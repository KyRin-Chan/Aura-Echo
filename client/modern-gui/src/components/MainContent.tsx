import { useState, useEffect } from 'react';
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors, DragEndEvent } from '@dnd-kit/core';
import { arrayMove, SortableContext } from '@dnd-kit/sortable';
import SortableCardItem from './Helpers/SortableCardItem';
import ModelSettingsCard from './ModelSettings/ModelSettingsCard';
import PerformanceStatsCard from './PerformanceStats/PerformanceStatsCard';
import AiSettingsCard from './AiSettings/AiSettingsCard';
import AudioSettingsCard from './AudioSettings/AudioSettingsCard';
import AudioEffectsCard from './AudioEffects/AudioEffectsCard';

// Define Card IDs
const CARD_IDS = {
  MODEL_SETTINGS: 'modelSettings',
  PERFORMANCE: 'performance',
  AI_SETTINGS: 'aiSettings',
  AUDIO_SETTINGS: 'audioSettings',
  AUDIO_EFFECTS: 'audioEffects',
};

const DEFAULT_CARD_ORDER: string[] = [
  CARD_IDS.MODEL_SETTINGS,
  CARD_IDS.PERFORMANCE,
  CARD_IDS.AI_SETTINGS,
  CARD_IDS.AUDIO_SETTINGS,
  CARD_IDS.AUDIO_EFFECTS,
];

const CARD_ORDER_STORAGE_KEY = 'aura_card_order';

function loadInitialCardOrder(): string[] {
  try {
    const raw = localStorage.getItem(CARD_ORDER_STORAGE_KEY);
    if (!raw) return DEFAULT_CARD_ORDER;
    const parsed = JSON.parse(raw);
    if (
      Array.isArray(parsed) &&
      parsed.length === DEFAULT_CARD_ORDER.length &&
      DEFAULT_CARD_ORDER.every((id) => parsed.includes(id))
    ) {
      return parsed;
    }
  } catch {
    // fallback to default order
  }
  return DEFAULT_CARD_ORDER;
}

function MainContent() {
  // ---------------- States ----------------
  const [cardOrder, setCardOrder] = useState<string[]>(loadInitialCardOrder);

  useEffect(() => {
    try {
      localStorage.setItem(CARD_ORDER_STORAGE_KEY, JSON.stringify(cardOrder));
    } catch {
      // ignore storage quota errors
    }
  }, [cardOrder]);

  // ---------------- Hooks ----------------

  // Create drag-and-drop sensor for the cards
  const sensors = useSensors(
    useSensor(PointerSensor)
  );

  // ---------------- Functions ----------------

  // Check if a card is wide (spans 2 columns)
  const isWideCard = (cardId: string) => {
    return cardId === CARD_IDS.AUDIO_EFFECTS;
  };

  // Handle drag end and updates position index of card
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      setCardOrder((items) => {
        const oldIndex = items.indexOf(active.id as string);
        let newIndex = items.indexOf(over.id as string);
        
        // Smart positioning for wide cards (2-unit tiles)
        if (isWideCard(active.id as string)) {
          // If trying to place wide card at an odd position, move to next even position
          if (newIndex % 2 === 1) {
            newIndex = Math.min(newIndex + 1, items.length);
          }
        } else {         
          // Check if there's a wide card that would be affected
          const reorderedItems = arrayMove(items, oldIndex, newIndex);
          
          // Adjust positions to maintain proper layout
          for (let i = 0; i < reorderedItems.length; i++) {
            if (isWideCard(reorderedItems[i]) && i % 2 === 1) {
              // Move wide card to next even position
              const wideCardIndex = i;
              const adjustedIndex = Math.min(i + 1, reorderedItems.length - 1);
              if (adjustedIndex !== wideCardIndex) {
                const wideCard = reorderedItems.splice(wideCardIndex, 1)[0];
                reorderedItems.splice(adjustedIndex, 0, wideCard);
              }
            }
          }
          
          return reorderedItems;
        }
        
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  // Generate grid items with proper positioning
  const generateGridItems = () => {
    const items: JSX.Element[] = [];
    let currentRow = 0;
    let currentCol = 0;

    cardOrder.forEach((cardId) => {
      const isWide = isWideCard(cardId);
      
      // If it's a wide card and we're in the second column, move to next row
      if (isWide && currentCol === 1) {
        currentRow++;
        currentCol = 0;
      }
      
      // If it's a normal card and there's no space in current row, move to next row
      if (!isWide && currentCol >= 2) {
        currentRow++;
        currentCol = 0;
      }

      let cardComponent;
      const gridClasses = isWide ? 'col-span-1 xl:col-span-2' : '';
      
      switch (cardId) {
        case 'modelSettings':
          cardComponent = (
            <SortableCardItem key={cardId} id={cardId} className={gridClasses}>
              {(attributes, listeners) => (
                <ModelSettingsCard dndAttributes={attributes} dndListeners={listeners} />
              )}
            </SortableCardItem>
          );
          break;
        case 'performance':
          cardComponent = (
            <SortableCardItem key={cardId} id={cardId} className={gridClasses}>
              {(attributes, listeners) => (
                <PerformanceStatsCard dndAttributes={attributes} dndListeners={listeners} />
              )}
            </SortableCardItem>
          );
          break;
        case 'aiSettings':
          cardComponent = (
            <SortableCardItem key={cardId} id={cardId} className={gridClasses}>
              {(attributes, listeners) => (
                <AiSettingsCard dndAttributes={attributes} dndListeners={listeners} />
              )}
            </SortableCardItem>
          );
          break;
        case 'audioSettings':
          cardComponent = (
            <SortableCardItem key={cardId} id={cardId} className={gridClasses}>
              {(attributes, listeners) => (
                <AudioSettingsCard dndAttributes={attributes} dndListeners={listeners} />
              )}
            </SortableCardItem>
          );
          break;
        case 'audioEffects':
          cardComponent = (
            <SortableCardItem key={cardId} id={cardId} className={gridClasses}>
              {(attributes, listeners) => (
                <AudioEffectsCard dndAttributes={attributes} dndListeners={listeners} />
              )}
            </SortableCardItem>
          );
          break;
        default:
          cardComponent = null;
      }
      
      if (cardComponent) {
        items.push(cardComponent);
        
        // Update position for next card
        if (isWide) {
          currentRow++;
          currentCol = 0;
        } else {
          currentCol++;
          if (currentCol >= 2) {
            currentRow++;
            currentCol = 0;
          }
        }
      }
    });
    
    return items;
  };

  // ---------------- Render ----------------

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragEnd={handleDragEnd}
    >
      <SortableContext
        items={cardOrder}
      >
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 auto-rows-min">
          {generateGridItems()}
        </div>
      </SortableContext>
    </DndContext>
  );
}

export default MainContent;
