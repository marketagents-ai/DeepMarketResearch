import logging
import time
import random
from typing import List, Tuple
from urllib.parse import quote_plus
import asyncio
from googlesearch import search
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

class SearchManager:
    # Global rate limiting parameters
    _last_request_time = 0
    _request_lock = asyncio.Lock()
    _min_delay = 15.0  # Increased delay to 15 seconds for more headroom
    _semaphore = asyncio.Semaphore(1)  # Only one active search at a time

    def __init__(self, config, worker_count: int = 1):
        self.config = config
        self.max_retries = 3
        self.headers = config.headers.to_dict() if hasattr(config, 'headers') else {}
        self.query_url_mapping = {}
        self.user_agent = UserAgent()
        # Centralized queue for search requests from all agents
        self._queue: asyncio.Queue[Tuple[str, int, asyncio.Future]] = asyncio.Queue()
        # Create worker(s) to process the queued searches sequentially
        self._workers = [asyncio.create_task(self._worker()) for _ in range(worker_count)]

    async def _worker(self):
        while True:
            query, num_results, future = await self._queue.get()
            try:
                result = await self._execute_search(query, num_results)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self._queue.task_done()

    async def _execute_search(self, query: str, num_results: int) -> List[str]:
        sanitized_query = quote_plus(query)
        async with self._semaphore:  # Ensure one search request at a time
            for attempt in range(self.max_retries):
                try:
                    async with self._request_lock:
                        now = time.time()
                        delay = self._min_delay - (now - self._last_request_time)
                        if delay > 0:
                            # Extra jitter to avoid predictable timing
                            jitter = random.uniform(1.0, 2.0)
                            total_delay = delay + jitter
                            logger.info(f"Rate limiting: waiting {total_delay:.2f}s for query: {query[:50]}...")
                            await asyncio.sleep(total_delay)
                        # Rotate user agent for each request
                        self.headers['User-Agent'] = self.user_agent.random
                        # Execute the search with a longer sleep_interval between internal calls
                        urls = list(search(
                            term=sanitized_query,
                            num_results=num_results,
                            lang="en",
                            sleep_interval=5,
                            timeout=30,
                            safe="active",
                            unique=True
                        ))
                        self._last_request_time = time.time()
                    if urls:
                        logger.info(f"Search successful for query: {query[:50]}... Found {len(urls)} URLs")
                        for url in urls:
                            self.query_url_mapping[url] = query
                        return urls
                except Exception as e:
                    logger.error(f"Search attempt {attempt+1}/{self.max_retries} failed for query: {query[:50]}... {str(e)}")
                    if attempt < self.max_retries - 1:
                        retry_delay = min(60, (2 ** attempt) + random.uniform(2, 5))
                        logger.info(f"Retrying in {retry_delay:.2f} seconds...")
                        await asyncio.sleep(retry_delay)
            logger.error(f"All search attempts failed for query: {query[:50]}...")
            return []

    async def get_urls_for_query(self, query: str, num_results: int = 2) -> List[str]:
        """
        Enqueues the query so that all agents share a single rate-limited search pool.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((query, num_results, future))
        return await future

    def reset(self):
        self.query_url_mapping.clear()