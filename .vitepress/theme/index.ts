import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h } from 'vue'

import './custom.css'
import ThreeDemo from './components/ThreeDemo.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {})
  },
  enhanceApp({ app }) {
    app.component('ThreeDemo', ThreeDemo)
  }
} satisfies Theme
