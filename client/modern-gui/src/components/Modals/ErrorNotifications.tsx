import React from 'react';

type ErrorType = 'Error' | 'Warning' | 'Confirm';

export interface UIError {
  id: number;
  message: string;
  type: ErrorType;
}

interface ErrorNotificationsProps {
  errors: UIError[];
  removeError: (id: number) => void;
}

const getNotificationStyle = (type: ErrorType) => {
  switch (type) {
    case 'Error':
      return { backgroundColor: 'var(--macaron-coral)', borderColor: 'var(--macaron-coral)', color: '#3a3530' };
    case 'Warning':
      return { backgroundColor: 'var(--macaron-yellow)', borderColor: 'var(--macaron-yellow)', color: '#3a3530' };
    case 'Confirm':
      return { backgroundColor: 'var(--macaron-mint)', borderColor: 'var(--macaron-mint)', color: '#3a3530' };
  }
};

const ErrorNotifications: React.FC<ErrorNotificationsProps> = ({ errors, removeError }) => (

  // ---------------- Render ----------------


  <div className="fixed top-4 right-4 space-y-2 z-50">
    {errors.map(err => (
      <div
        key={err.id}
        className="max-w-sm w-full p-4 rounded-2xl shadow-lg flex justify-between items-start"
        style={{
          ...getNotificationStyle(err.type),
          border: `1px solid`,
        }}
      >
        <div>
          <strong className="block text-sm font-semibold">{err.type}</strong>
          <p className="text-sm">{err.message}</p>
        </div>
        <button
          onClick={() => removeError(err.id)}
          className="ml-4 text-lg font-bold leading-none opacity-70 hover:opacity-100 transition-opacity"
        >
          &times;
        </button>
      </div>
    ))}
  </div>
);

export default ErrorNotifications;
