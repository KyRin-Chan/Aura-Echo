import { JSX, useMemo, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPlus,
  faTrash,
  faGripVertical,
  faVolumeUp,
  faMicrophone,
  faVolumeHigh,
  faCog
} from '@fortawesome/free-solid-svg-icons';
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors, DragEndEvent } from '@dnd-kit/core';
import { arrayMove, SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { AudioEffect, AudioChannel } from '@dannadori/voice-changer-client-js';
import { getEffectDefinition } from './serverEffectsUtils';
import AddEffectModal from './AddEffectModal';
import { CSS_CLASSES } from '../../styles/constants';
import { t } from '../../locales';

export type AudioEffectWithIndex = AudioEffect & { index: number };

type Props = {
  channel: AudioChannel;
  effects: AudioEffectWithIndex[];
  selectedEffectIndex: number | null;
  onEffectSelect: (effectIndex: number) => void;
  onEffectAdd: (effectType: string, channel: AudioChannel) => void;
  onEffectDelete: (effectIndex: number) => void;
  onEffectToggle: (effectIndex: number) => void;
  onEffectReorder: (reorderedChannelEffects: AudioEffectWithIndex[]) => void;
  serverSchema?: any;
  providersInfo?: any;
};

function SortableEffectItem({
  effect,
  isSelected,
  onSelect,
  onDelete,
  onToggle,
  serverSchema
}: {
  effect: AudioEffectWithIndex;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onToggle: () => void;
  serverSchema?: any;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: effect.index.toString()
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`p-3 rounded-md border cursor-pointer transition-all duration-150 ${
        isSelected
          ? 'border-primary bg-primary/8 shadow-elevation-1'
          : 'border-outline-variant bg-surface-container-low hover:border-outline'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="flex items-center justify-center w-6 h-6 bg-primary-container text-on-primary-container rounded-full text-xs font-semibold">
            {effect.index + 1}
          </div>
          <button
            {...attributes}
            {...listeners}
            className={`${CSS_CLASSES.iconButton} cursor-grab active:cursor-grabbing`}
            title={t('dragToReorder')}
            onClick={(e) => e.stopPropagation()}
          >
            <FontAwesomeIcon icon={faGripVertical} className="h-4 w-4" />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggle();
            }}
            className={`${CSS_CLASSES.iconButton} ${
              effect.enabled ? 'text-primary' : 'text-on-surface-variant/40'
            }`}
            title={effect.enabled ? t('disableTooltip') : t('enableTooltip')}
          >
            <FontAwesomeIcon icon={faVolumeUp} className={`h-4 w-4 ${effect.enabled ? '' : 'opacity-40'}`} />
          </button>
          <div>
            <div className="font-semibold text-on-surface text-sm">
              {getEffectDefinition(effect.type, serverSchema)?.name || effect.type}
            </div>
            <div className="text-xs text-on-surface-variant flex items-center space-x-2">
              <span className="capitalize">{effect.type}</span>
              {getEffectDefinition(effect.type, serverSchema)?.provider && (
                <span className="px-2 py-0.5 bg-surface-container-highest text-on-surface-variant rounded-full text-[10px] font-medium">
                  {getEffectDefinition(effect.type, serverSchema)?.provider}
                </span>
              )}
            </div>
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className={`${CSS_CLASSES.iconButton} text-error hover:text-error/80`}
          title={t('deleteTooltip')}
        >
          <FontAwesomeIcon icon={faTrash} className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

export default function EffectsList({
  channel,
  effects,
  selectedEffectIndex,
  onEffectSelect,
  onEffectAdd,
  onEffectDelete,
  onEffectToggle,
  onEffectReorder,
  serverSchema,
  providersInfo
}: Props): JSX.Element {
  const [showAddModal, setShowAddModal] = useState(false);
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }));
  const channelEffects = useMemo(
    () => (effects || []).slice().sort((a, b) => a.index - b.index),
    [effects]
  );
  const memoizedSortableItems = useMemo(() => channelEffects.map((e) => e.index.toString()), [channelEffects]);

  const handleAddEffect = (effectType: string) => {
    onEffectAdd(effectType, channel);
    setShowAddModal(false);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id || !channelEffects.length) return;
    const oldIndex = channelEffects.findIndex((e) => e.index.toString() === active.id);
    const newIndex = channelEffects.findIndex((e) => e.index.toString() === over.id);
    if (oldIndex !== -1 && newIndex !== -1) {
      const reordered = arrayMove(channelEffects, oldIndex, newIndex);
      onEffectReorder(reordered);
    }
  };

  return (
    <div className="flex flex-col h-full bg-surface-container-low p-4 rounded-md border border-outline-variant">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-outline-variant">
        <div>
          <h5 className="font-semibold text-on-surface text-base">
            {channel === 'input' ? t('inputChain') : t('outputChain')}
          </h5>
          <div className="text-xs text-on-surface-variant mt-1">{t('signalFlowDesc')}</div>
        </div>
        <div className="relative">
          <button
            onClick={() => setShowAddModal(true)}
            className={`${CSS_CLASSES.iconButton} text-primary`}
            title={t('addEffect')}
          >
            <FontAwesomeIcon icon={faPlus} className="h-4 w-4" />
          </button>
        </div>
      </div>

      {channelEffects.length === 0 ? (
        <div className="text-center py-12 text-on-surface-variant/60 text-sm italic">
          {channel === 'input' ? t('noInputEffects') : t('noOutputEffects')}
          <br />
          {t('clickPlusToAddEffect')}
        </div>
      ) : (
        <div className="space-y-1 overflow-y-auto pr-1">
          <div className="flex justify-center py-1.5">
            <div className="flex items-center space-x-2 px-3 py-1.5 bg-surface-container-highest rounded-full border border-outline-variant/30">
              <FontAwesomeIcon
                icon={channel === 'input' ? faMicrophone : faCog}
                className={`h-3.5 w-3.5 ${channel === 'input' ? 'text-primary' : 'text-secondary'}`}
              />
              <span className="text-xs text-on-surface-variant font-semibold">
                {channel === 'input' ? t('audioInputTitle') : t('fromProcessing')}
              </span>
            </div>
          </div>

          <div className="flex justify-center py-1">
            <div className="w-0.5 h-4 bg-outline-variant"></div>
          </div>

          <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
            <SortableContext items={memoizedSortableItems} strategy={verticalListSortingStrategy}>
              <div className="space-y-1">
                {channelEffects.map((effect, channelIndex) => (
                  <div key={effect.index}>
                    <SortableEffectItem
                      effect={effect}
                      isSelected={selectedEffectIndex === effect.index}
                      onSelect={() => onEffectSelect(effect.index)}
                      onDelete={() => onEffectDelete(effect.index)}
                      onToggle={() => onEffectToggle(effect.index)}
                      serverSchema={serverSchema}
                    />
                    {channelIndex < channelEffects.length - 1 && (
                      <div className="flex justify-center py-1">
                        <div className="w-0.5 h-4 bg-outline-variant"></div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </SortableContext>
          </DndContext>

          <div className="flex justify-center py-1">
            <div className="w-0.5 h-4 bg-outline-variant"></div>
          </div>

          <div className="flex justify-center py-1.5">
            <div className="flex items-center space-x-2 px-3 py-1.5 bg-surface-container-highest rounded-full border border-outline-variant/30">
              <FontAwesomeIcon
                icon={channel === 'input' ? faCog : faVolumeHigh}
                className={`h-3.5 w-3.5 ${channel === 'input' ? 'text-primary' : 'text-secondary'}`}
              />
              <span className="text-xs text-on-surface-variant font-semibold">
                {channel === 'input' ? t('toProcessing') : t('audioOutputTitle')}
              </span>
            </div>
          </div>
        </div>
      )}

      <AddEffectModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onAddEffect={(type) => handleAddEffect(type)}
        channel={channel}
        serverSchema={serverSchema}
        providersInfo={providersInfo}
      />
    </div>
  );
}
