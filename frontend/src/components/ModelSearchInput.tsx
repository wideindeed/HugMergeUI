import { useEffect, useRef, useState } from 'react'
import { searchModels } from '../api/client'

interface Props {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
}

export function ModelSearchInput({ id, label, value, onChange }: Props) {
  const [suggestions, setSuggestions] = useState<string[]>([])
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(() => {
    if (value.trim().length < 2) {
      setSuggestions([])
      return
    }
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      searchModels(value)
        .then(setSuggestions)
        .catch(() => setSuggestions([]))
    }, 300)
    return () => clearTimeout(debounceRef.current)
  }, [value])

  const listId = `${id}-suggestions`

  return (
    <label className="model-search-input">
      {label}
      <input
        type="text"
        list={listId}
        value={value}
        placeholder="org/model-name"
        onChange={(e) => onChange(e.target.value)}
        autoComplete="off"
        spellCheck={false}
      />
      <datalist id={listId}>
        {suggestions.map((s) => (
          <option key={s} value={s} />
        ))}
      </datalist>
    </label>
  )
}
