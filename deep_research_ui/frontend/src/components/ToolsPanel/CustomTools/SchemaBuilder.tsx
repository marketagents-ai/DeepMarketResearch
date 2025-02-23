import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import type { CustomTool, SchemaField } from '../../../types/tools';

interface SchemaBuilderProps {
  initialTool?: CustomTool | null;
  onCancel: () => void;
  onSave: (tool: CustomTool) => void;
}

export const SchemaBuilder: React.FC<SchemaBuilderProps> = ({ initialTool, onCancel, onSave }) => {
  const [name, setName] = useState(initialTool?.name || '');
  const [description, setDescription] = useState(initialTool?.description || '');
  const [fields, setFields] = useState<SchemaField[]>(
    initialTool?.schema?.properties 
      ? Object.entries(initialTool.schema.properties).map(([name, field]: [string, any]) => ({
          name,
          type: field.type,
          required: field.required || false
        }))
      : []
  );

  const addField = () => {
    setFields([...fields, { name: '', type: 'string', required: false }]);
  };

  const updateField = (index: number, updates: Partial<SchemaField>) => {
    setFields(fields.map((field, i) => 
      i === index ? { ...field, ...updates } : field
    ));
  };

  const removeField = (index: number) => {
    setFields(fields.filter((_, i) => i !== index));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({
      id: initialTool?.id || '',
      name,
      description,
      type: "custom",
      enabled: initialTool?.enabled ?? true,
      schema: {
        type: 'object',
        properties: Object.fromEntries(
          fields.map(field => [
            field.name,
            { 
              type: field.type, 
              required: field.required,
              description: `Analysis and insights related to ${field.name} in financial markets` // Add description
            }
          ])
        )
      }
    });
  };


  return (
    <form onSubmit={handleSubmit} className="bg-gray-700 rounded-lg p-4 mb-4">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full bg-gray-600 rounded px-3 py-1.5 text-sm"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Description</label>
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full bg-gray-600 rounded px-3 py-1.5 text-sm"
            required
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium">Schema Fields</label>
            <button
              type="button"
              onClick={addField}
              className="text-xs bg-blue-600 px-2 py-1 rounded"
            >
              Add Field
            </button>
          </div>

          <div className="space-y-2">
            {fields.map((field, index) => (
              <div key={index} className="flex gap-2 items-start">
                <input
                  type="text"
                  value={field.name}
                  onChange={(e) => updateField(index, { name: e.target.value })}
                  placeholder="Field name"
                  className="flex-1 bg-gray-600 rounded px-2 py-1 text-sm"
                />
                <select
                  value={field.type}
                  onChange={(e) => updateField(index, { type: e.target.value as any })}
                  className="bg-gray-600 rounded px-2 py-1 text-sm"
                >
                  <option value="string">String</option>
                  <option value="number">Number</option>
                  <option value="boolean">Boolean</option>
                  <option value="array">Array</option>
                </select>
                <button
                  type="button"
                  onClick={() => removeField(index)}
                  className="p-1 hover:bg-gray-600 rounded"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-4">
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1.5 text-sm hover:bg-gray-600 rounded"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 rounded"
          >
            Save Tool
          </button>
        </div>
      </div>
    </form>
  );
};