import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowsAlt } from '@fortawesome/free-solid-svg-icons';

interface DragHandleProps {
  // Pass attributes and listeners from useSortable down to this handle
  attributes?: Record<string, any>;
  listeners?: Record<string, any>;
  className?: string;
  title?: string;
}

const DragHandle: React.FC<DragHandleProps> = ({ attributes, listeners, className, title = "Drag" }) => {
  return (
    <button 
      {...attributes} 
      {...listeners} 
      className={`p-1 cursor-grab transition-opacity hover:opacity-80 ${className}`}
      style={{ color: 'var(--text-tertiary)' }}
      title={title}
    >
      <FontAwesomeIcon icon={faArrowsAlt} className="h-5 w-5" />
    </button>
  );
};

export default DragHandle; 