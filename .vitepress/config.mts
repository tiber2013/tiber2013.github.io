import { defineConfig } from 'vitepress'
import { readingNotesSidebar } from './readingnotes.mts'

export default defineConfig({
  base: '/',
  lang: 'en-US',
  title: 'Tiber Blog',
  description: 'AI Engineer - Computer Vision & Deep Learning',
  lastUpdated: true,
  cleanUrls: true,
  srcExclude: ['readingnotes/**/_source/**'],
  markdown: {
    math: true,
  },
  head: [
    ['meta', { name: 'theme-color', content: '#6366f1' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'en-US' }],
  ],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'Tiber Blog',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Paper Notes', link: '/notes/' },
      { text: 'Reading Notes', link: '/readingnotes/' },
      { text: 'Projects', link: '/projects/' },
      { text: 'Resume', link: '/resume/' },
    ],
    sidebar: {
      '/readingnotes/': readingNotesSidebar,
      '/notes/': [
        {
          text: 'Paper Notes',
          items: [
            { text: 'Overview', link: '/notes/' },
            { text: 'Tags Index', link: '/notes/tags' },
          ]
        },
        {
          text: 'Object Detection',
          collapsed: false,
          items: [
            { text: 'YOLO Series', link: '/notes/detection/yolo' },
          ]
        },
        {
          text: 'Image Segmentation',
          collapsed: false,
          items: [
            { text: 'SAM Series', link: '/notes/segmentation/sam' },
          ]
        },
        {
          text: 'Transformer',
          collapsed: false,
          items: [
            { text: 'Vision Transformer', link: '/notes/transformer/vit' },
          ]
        },
      ],
      '/projects/': [
        {
          text: 'Projects',
          items: [
            { text: 'Overview', link: '/projects/' },
            { text: 'Real-time Detection', link: '/projects/detection' },
            { text: '3D Demo', link: '/projects/3d-demo' },
          ]
        },
      ],
      '/resume/': [
        {
          text: 'Resume',
          items: [
            { text: 'Chinese Resume', link: '/resume/' },
            { text: 'English Resume', link: '/resume/en' },
          ]
        },
      ],
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/tiber2013' }
    ],
    footer: {
      message: 'Built with VitePress',
      copyright: 'Copyright © 2024-present Tiber'
    },
    search: {
      provider: 'local'
    },
    outline: {
      level: [2, 3]
    },
    docFooter: {
      prev: 'Previous',
      next: 'Next'
    },
    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },
  },
})
