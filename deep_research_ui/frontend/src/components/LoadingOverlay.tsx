export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ isLoading, message }) => {
    if (!isLoading) return null;
  
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg p-8 shadow-xl max-w-md w-full mx-4">
          <LoadingSpinner />
          {message && (
            <div className="mt-4 text-center text-gray-300">
              {message}
            </div>
          )}
        </div>
      </div>
    );
  };