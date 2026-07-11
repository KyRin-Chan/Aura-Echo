import { RVCModelSlot } from "@dannadori/voice-changer-client-js";
import { faPen, faTrash } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useInitialPlaceholder } from "../../scripts/usePlaceholder";
import { useState } from "react";
import DeleteModelModal from "./Modals/DeleteModelModal";
import EditModelModal from "./Modals/EditModelModal";

interface ModelSlotProps {
  selected: boolean;
  model: RVCModelSlot;
  modelDir: string;
  handleSelectModel: (model: RVCModelSlot) => void;
}

function ModelSlot(props: ModelSlotProps) {
  // ---------------- State ----------------
  const [showEdit, setShowEdit] = useState<boolean>(false);
  const [showDelete, setShowDelete] = useState<boolean>(false);

  // Generate icon URL from model directory and icon file path (or placeholder if not existing)
  const icon = props.model.iconFile.length > 0 ? "/" + props.modelDir + "/" + props.model.slotIndex + "/" + props.model.iconFile.split(/[\/\\]/).pop() : "";
  const placeholder = useInitialPlaceholder(props.model.name);

  // ---------------- Render ----------------

  return (
    <>
      {/* Edit modal for updating model settings */}
      <EditModelModal
        model={props.model}
        showModal={showEdit}
        setShowEdit={setShowEdit}
        modelDir={props.modelDir}
      />
      {/* Delete confirmation modal */}
      <DeleteModelModal
        model={props.model}
        showModal={showDelete}
        setShowDelete={setShowDelete}
        modelDir={props.modelDir}
      />
      {/* Main model slot container with conditional styling for selection state */}
      <li
        className={`p-3 text-xs cursor-pointer flex items-center group transition-all duration-150 rounded-full mx-1.5 my-1 ${
          props.selected
            ? 'bg-primary text-on-primary font-bold shadow-elevation-1'
            : 'bg-transparent text-on-surface hover:bg-surface-variant/20'
        }`}
        onClick={() => {
          props.handleSelectModel(props.model);
        }}
      >
        {/* Model thumbnail with fallback to generated placeholder */}
        <img
          src={icon.length > 0 ? icon : placeholder}
          alt={props.model.name}
          className="w-8 h-8 rounded-full mr-2.5 object-cover flex-shrink-0 border border-outline-variant/30"
        />
        {/* Model name with text truncation for long names */}
        <span className="truncate mr-2 flex-grow">{props.model.name}</span>
        <div className="flex space-x-1 items-center opacity-0 group-hover:opacity-100 transition-opacity">
          {/* Edit button - Opens model settings modal */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowEdit(true);
            }}
            className={`p-1.5 rounded-full transition-all hover:bg-surface-variant/30 ${
              props.selected ? 'text-on-primary' : 'text-secondary'
            }`}
            title="Edit Model"
          >
            <FontAwesomeIcon icon={faPen} className="h-3 w-3" />
          </button>
          {/* Delete button - Opens confirmation modal */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowDelete(true);
            }}
            className={`p-1.5 rounded-full transition-all hover:bg-surface-variant/30 ${
              props.selected ? 'text-on-primary' : 'text-error'
            }`}
            title="Delete Model"
          >
            <FontAwesomeIcon icon={faTrash} className="h-3 w-3" />
          </button>
        </div>
      </li>
    </>
  );
}

export default ModelSlot;
