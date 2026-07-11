import { JSX, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTimes } from '@fortawesome/free-solid-svg-icons';
import { CSS_CLASSES } from '../../styles/constants';

interface ModalButton {
  text: string;
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}

type ModalSize = 'small' | 'medium' | 'large';

interface GenericModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  primaryButton?: ModalButton;
  secondaryButton?: ModalButton;
  transparent?: boolean;
  closeOnOutsideClick?: boolean;
  size?: ModalSize;
}

function GenericModal({
  isOpen,
  onClose,
  title,
  children,
  primaryButton,
  secondaryButton,
  transparent = false,
  closeOnOutsideClick = true,
  size = 'medium'
}: GenericModalProps): JSX.Element | null {
  if (!isOpen) {
    return null;
  }

  const modalRoot = document.getElementById('modal-root');
  if (!modalRoot) {
    return null;
  }

  // Define modal widths based on size
  const getModalWidth = (size: ModalSize): string => {
    switch (size) {
      case 'small':
        return 'max-w-lg';
      case 'medium':
        return 'max-w-4xl';
      case 'large':
        return 'max-w-7xl';
      default:
        return 'max-w-4xl';
    }
  };

  return createPortal(
    <div
      className={`fixed inset-0 ${
        transparent === false ? 'bg-black bg-opacity-50 backdrop-blur-sm' : 'opacity-100'
      } flex justify-center items-center z-40 p-4 transition-opacity duration-300 ease-in-out`}
      onClick={closeOnOutsideClick ? onClose : undefined}
    >
      <div
        className={`p-6 rounded-xl shadow-elevation-3 w-full ${getModalWidth(
          size
        )} max-h-[90vh] flex flex-col transform transition-all duration-300 ease-in-out scale-95 opacity-0 animate-modalFadeInScaleUp overflow-hidden bg-surface-container-highest border border-outline-variant`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex justify-between items-center mb-4 pb-3 border-b border-outline-variant">
          <h3 className="text-xl font-semibold text-on-surface">{title}</h3>
          <button
            onClick={onClose}
            className="p-1 rounded-full focus:outline-none focus:ring-2 focus:ring-primary transition-all text-on-surface-variant hover:bg-surface-variant/20"
            aria-label="Close modal"
          >
            <FontAwesomeIcon icon={faTimes} className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="overflow-y-auto mb-6 flex-grow text-on-surface">
          {children}
        </div>

        {/* Modal Footer (optional buttons) */}
        {(primaryButton || secondaryButton) && (
          <div className="flex justify-end space-x-3 pt-3 border-t border-outline-variant">
            {secondaryButton && (
              <button
                onClick={secondaryButton.onClick}
                className={`${CSS_CLASSES.modalSecondaryButton} ${secondaryButton.className || ''} ${
                  secondaryButton.disabled ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                disabled={secondaryButton.disabled}
              >
                {secondaryButton.text}
              </button>
            )}
            {primaryButton && (
              <button
                onClick={primaryButton.onClick}
                className={`${CSS_CLASSES.modalPrimaryButton} ${primaryButton.className || ''} ${
                  primaryButton.disabled ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                disabled={primaryButton.disabled}
              >
                {primaryButton.text}
              </button>
            )}
          </div>
        )}
      </div>
    </div>,
    modalRoot
  );
}

export default GenericModal;