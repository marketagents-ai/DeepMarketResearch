import Papa from 'papaparse';
import * as XLSX from 'xlsx';

export const parseFile = async (file: File): Promise<any[]> => {
  const extension = file.name.split('.').pop()?.toLowerCase();

  if (!extension) {
    throw new Error('Invalid file type');
  }

  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = async (e) => {
      try {
        const result = e.target?.result;
        
        switch (extension) {
          case 'csv':
          case 'tsv': {
            Papa.parse(file, {
              complete: (results) => {
                resolve(results.data);
              },
              header: true,
              error: (error) => reject(error),
            });
            break;
          }
          case 'xlsx':
          case 'xls': {
            if (!result) throw new Error('Failed to read file');
            const workbook = XLSX.read(result, { type: 'binary' });
            const worksheet = workbook.Sheets[workbook.SheetNames[0]];
            const data = XLSX.utils.sheet_to_json(worksheet);
            resolve(data);
            break;
          }
          default:
            throw new Error('Unsupported file type');
        }
      } catch (error) {
        reject(error);
      }
    };

    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsBinaryString(file);
  });
};