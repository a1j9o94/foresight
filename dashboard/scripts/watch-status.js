#!/usr/bin/env node
/**
 * Watch for changes to results files and regenerate status.json
 * Run this alongside the dev server for live updates
 */

import chokidar from 'chokidar';
import { execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '../..');
const DASHBOARD_DIR = path.resolve(__dirname, '..');

// Watch patterns
const watchPaths = [
  path.join(ROOT, 'research/experiments/*/results.yaml'),
  path.join(ROOT, 'research/experiments/*/FINDINGS.md'),
];

let debounceTimer = null;
let regenerating = false;

function regenerate() {
  if (regenerating) return;
  regenerating = true;

  console.log('\n\x1b[34mRegenrating status.json...\x1b[0m');
  try {
    execSync('node scripts/generate-status.js', {
      cwd: DASHBOARD_DIR,
      stdio: 'inherit'
    });
    console.log('\x1b[32mStatus updated!\x1b[0m (dashboard will auto-refresh)');
  } catch (e) {
    console.error('\x1b[31mError regenerating status:\x1b[0m', e.message);
  }
  regenerating = false;
}

// Initial generation
regenerate();

const watcher = chokidar.watch(watchPaths, {
  persistent: true,
  ignoreInitial: true,
  awaitWriteFinish: {
    stabilityThreshold: 300,
    pollInterval: 100
  }
});

watcher
  .on('change', (filepath) => {
    const relative = path.relative(ROOT, filepath);
    console.log(`\x1b[90mChanged: ${relative}\x1b[0m`);
    // Debounce to avoid multiple regenerations
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(regenerate, 500);
  })
  .on('add', (filepath) => {
    const relative = path.relative(ROOT, filepath);
    console.log(`\x1b[90mAdded: ${relative}\x1b[0m`);
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(regenerate, 500);
  })
  .on('error', (error) => {
    console.error('\x1b[31mWatcher error:\x1b[0m', error);
  });

console.log('\n\x1b[1m═══════════════════════════════════════════════════════════\x1b[0m');
console.log('\x1b[1m  STATUS WATCHER\x1b[0m');
console.log('\x1b[1m═══════════════════════════════════════════════════════════\x1b[0m');
console.log('\nWatching for changes to experiment results...');
console.log('  - research/experiments/*/results.yaml');
console.log('  - research/experiments/*/FINDINGS.md');
console.log('\nPress Ctrl+C to stop.\n');
