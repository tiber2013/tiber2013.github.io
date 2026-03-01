<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import * as THREE from 'three'

const container = ref<HTMLDivElement>()
let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let animationId: number

const props = defineProps<{
  type?: 'cube' | 'sphere' | 'custom'
  width?: number
  height?: number
}>()

onMounted(() => {
  if (!container.value) return

  const width = props.width || container.value.clientWidth
  const height = props.height || 400

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x1a1a2e)

  camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
  camera.position.z = 3

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(width, height)
  renderer.setPixelRatio(window.devicePixelRatio)
  container.value.appendChild(renderer.domElement)

  const geometry = props.type === 'sphere' 
    ? new THREE.SphereGeometry(1, 32, 32)
    : new THREE.BoxGeometry(1.5, 1.5, 1.5)
  
  const material = new THREE.MeshNormalMaterial()
  const mesh = new THREE.Mesh(geometry, material)
  scene.add(mesh)

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
  scene.add(ambientLight)

  const pointLight = new THREE.PointLight(0xffffff, 1)
  pointLight.position.set(5, 5, 5)
  scene.add(pointLight)

  function animate() {
    animationId = requestAnimationFrame(animate)
    mesh.rotation.x += 0.01
    mesh.rotation.y += 0.01
    renderer.render(scene, camera)
  }
  animate()
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  if (renderer) {
    renderer.dispose()
  }
})
</script>

<template>
  <div ref="container" class="three-demo" :style="{ height: (height || 400) + 'px' }"></div>
</template>

<style scoped>
.three-demo {
  border-radius: 12px;
  overflow: hidden;
  margin: 16px 0;
  border: 1px solid var(--vp-c-divider);
}
</style>
