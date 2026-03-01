---
title: 3D Demo
---

# 3D Demo

Here's an interactive Three.js demo embedded in the page.

## Rotating Cube

<ThreeDemo type="cube" :height="300" />

## Rotating Sphere

<ThreeDemo type="sphere" :height="300" />

---

## How to Use

In any markdown file, simply use:

```markdown
<ThreeDemo type="cube" :height="300" />
<ThreeDemo type="sphere" :height="400" />
```

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| type | 'cube' \| 'sphere' | 'cube' | Shape type |
| height | number | 400 | Container height in px |

---

## Custom Three.js

For custom 3D scenes, create a new Vue component in `.vitepress/theme/components/` and register it in `index.ts`.
