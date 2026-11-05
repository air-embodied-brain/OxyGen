/// <reference types="astro/client" />

interface Window {
  oxygenCriticalFontsReady: Promise<void>;
  oxygenFontsReady: Promise<void>;
  oxygenAssetsReady: Promise<void>;
}
