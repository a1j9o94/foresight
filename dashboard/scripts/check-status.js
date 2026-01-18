#!/usr/bin/env node
/**
 * Check experiment status: regenerate data, print CLI summary, open dashboard
 */

import fs from 'fs';
import path from 'path';
import { execSync, spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Colors for terminal output
const colors = {
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  blue: '\x1b[34m',
  gray: '\x1b[90m',
  bold: '\x1b[1m',
  reset: '\x1b[0m',
};

function statusIcon(status, recommendation) {
  if (status === 'completed' && recommendation === 'proceed') return `${colors.green}✓${colors.reset}`;
  if (status === 'completed' && recommendation === 'pivot') return `${colors.yellow}⚠${colors.reset}`;
  if (status === 'completed') return `${colors.blue}●${colors.reset}`;
  if (status === 'in_progress' || status === 'running') return `${colors.yellow}◐${colors.reset}`;
  return `${colors.gray}○${colors.reset}`;
}

function gateIcon(status) {
  if (status === 'passed') return `${colors.green}✓${colors.reset}`;
  if (status === 'blocked') return `${colors.yellow}⚠${colors.reset}`;
  if (status === 'in_progress') return `${colors.yellow}◐${colors.reset}`;
  return `${colors.gray}○${colors.reset}`;
}

function printSummary(status) {
  console.log('\n' + colors.bold + '═══════════════════════════════════════════════════════════' + colors.reset);
  console.log(colors.bold + '  FORESIGHT EXPERIMENT STATUS' + colors.reset);
  console.log(colors.bold + '═══════════════════════════════════════════════════════════' + colors.reset + '\n');

  // Gates summary
  console.log(colors.bold + 'Decision Gates:' + colors.reset);
  for (const gate of status.gates) {
    const icon = gateIcon(gate.status);
    const statusText = gate.status === 'blocked' ? `BLOCKED` : gate.progress;
    console.log(`  ${icon} ${gate.name}: ${statusText}`);
    if (gate.status === 'blocked' && gate.blockers?.length > 0) {
      const reason = gate.blockerReason === 'failed'
        ? `${colors.red}↳ Code failures - fix and re-run: ${gate.blockers.join(', ')}${colors.reset}`
        : `${colors.yellow}↳ Hypothesis not supported (pivot): ${gate.blockers.join(', ')}${colors.reset}`;
      console.log(`      ${reason}`);
    }
  }

  console.log('\n' + colors.bold + 'Experiments:' + colors.reset);

  // Group by phase
  const phases = {};
  for (const exp of status.experiments) {
    if (!phases[exp.phase]) phases[exp.phase] = [];
    phases[exp.phase].push(exp);
  }

  // Check which experiments have active agents
  const activeExpIds = new Set(
    (status.agents || [])
      .filter(a => a.isActive)
      .map(a => a.experiment)
  );

  for (const [phase, exps] of Object.entries(phases)) {
    console.log(`\n  ${colors.gray}Phase ${phase}:${colors.reset}`);
    for (const exp of exps) {
      // Override status if there's an active agent working on this experiment
      const hasActiveAgent = activeExpIds.has(exp.id);
      const effectiveStatus = hasActiveAgent && exp.status === 'not_started' ? 'in_progress' : exp.status;

      const icon = statusIcon(effectiveStatus, exp.recommendation);
      let statusText = effectiveStatus === 'completed' && exp.recommendation === 'proceed' ? 'PASSED' :
                       effectiveStatus === 'completed' && exp.recommendation === 'pivot' ? 'PIVOT' :
                       effectiveStatus === 'completed' ? 'Complete' :
                       effectiveStatus === 'in_progress' ? 'Running' :
                       effectiveStatus === 'not_started' ? 'Not Started' : effectiveStatus;

      // Add key metrics if available
      let metrics = '';
      if (exp.metrics?.lpips !== undefined) {
        metrics = ` ${colors.gray}(LPIPS: ${exp.metrics.lpips})${colors.reset}`;
      }

      console.log(`    ${icon} ${exp.name}: ${statusText}${metrics}`);
    }
  }

  // Modal runs (from last hour)
  const agents = status.agents || [];
  if (agents.length > 0) {
    const running = agents.filter(a => a.state === 'running');
    const stopped = agents.filter(a => a.state === 'stopped');

    if (running.length > 0) {
      console.log('\n' + colors.bold + `Running on Modal (${running.length}):` + colors.reset);
      for (const agent of running) {
        console.log(`  ${colors.green}●${colors.reset} ${agent.experiment} ${colors.gray}(${agent.id.slice(0, 12)}...)${colors.reset}`);
      }
    }

    if (stopped.length > 0) {
      console.log('\n' + colors.bold + `Recent Runs (${stopped.length}):` + colors.reset);
      for (const agent of stopped) {
        console.log(`  ${colors.gray}○${colors.reset} ${agent.experiment} ${colors.gray}(stopped ${new Date(agent.stoppedAt).toLocaleTimeString()})${colors.reset}`);
      }
    }
  }

  console.log('\n' + colors.gray + `Last updated: ${new Date(status.generatedAt).toLocaleString()}` + colors.reset);
  console.log(colors.bold + '═══════════════════════════════════════════════════════════' + colors.reset + '\n');
}

async function main() {
  const dashboardDir = path.resolve(__dirname, '..');
  const projectRoot = path.resolve(dashboardDir, '..');
  const skipSync = process.argv.includes('--no-sync');

  // 1. Sync results from Modal (unless --no-sync flag)
  if (!skipSync) {
    console.log(`${colors.blue}Syncing results from Modal...${colors.reset}`);
    try {
      execSync('bash scripts/sync-results.sh', { cwd: projectRoot, stdio: 'inherit' });
    } catch (e) {
      console.log(`${colors.yellow}Warning: Could not sync from Modal (may not have results yet)${colors.reset}`);
    }
  }

  // 2. Regenerate status
  console.log(`${colors.blue}Regenerating status...${colors.reset}`);
  execSync('node scripts/generate-status.js', { cwd: dashboardDir, stdio: 'inherit' });

  // 2. Read and print summary
  const statusPath = path.join(dashboardDir, 'public/status.json');
  const status = JSON.parse(fs.readFileSync(statusPath, 'utf-8'));
  printSummary(status);

  // 3. Check if dev server is already running
  let serverRunning = false;
  try {
    const response = await fetch('http://localhost:5173/', { method: 'HEAD' });
    serverRunning = response.ok;
  } catch (e) {
    serverRunning = false;
  }

  // 4. Start dev server if not running
  if (!serverRunning) {
    console.log(`${colors.blue}Starting dev server...${colors.reset}`);
    const devServer = spawn('npm', ['run', 'dev'], {
      cwd: dashboardDir,
      detached: true,
      stdio: 'ignore',
    });
    devServer.unref();
    // Wait for server to start
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // 5. Open in Chrome
  console.log(`${colors.blue}Opening dashboard in Chrome...${colors.reset}`);
  const url = 'http://localhost:5173/';

  if (process.platform === 'darwin') {
    execSync(`open -a "Google Chrome" "${url}"`);
  } else if (process.platform === 'win32') {
    execSync(`start chrome "${url}"`);
  } else {
    execSync(`xdg-open "${url}"`);
  }

  console.log(`${colors.green}Done!${colors.reset} Dashboard: ${url}\n`);
}

main().catch(console.error);
