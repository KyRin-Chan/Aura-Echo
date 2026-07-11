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

const getNotificationClass = (type: ErrorType) => {
  switch (type) {
    case 'Error':
      return 'bg-error-container text-on-error-container border-error/20';
    case 'Warning':
      return 'bg-tertiary-container text-on-tertiary-container border-tertiary/20';
    case 'Confirm':
      return 'bg-primary-container text-on-primary-container border-primary/20';
  }
};

const ErrorNotifications: React.FC<ErrorNotificationsProps> = ({ errors, removeError }) => (
  <div className="fixed top-6 right-6 space-y-2 z-50 animate-fadeIn">
    {errors.map((err) => (
      <div
        key={err.id}
        className={`max-w-sm w-full p-4 rounded-2xl shadow-elevation-2 flex justify-between items-start border ${getNotificationClass(
          err.type
        )}`}
      >
        <div>
          <strong className="block text-xs uppercase tracking-wider font-bold mb-1 opacity-80">{err.type}</strong>
          <p className="text-xs font-semibold leading-relaxed">{err.message}</p>
        </div>
        <button
          onClick={() => removeError(err.id)}
          className="ml-4 text-base font-bold leading-none opacity-60 hover:opacity-100 transition-opacity"
        >
          &times;
        </button>
      </div>
    ))}
  </div>
);

export default ErrorNotifications;
