#!/usr/bin/env node
/**
 * Generates status.json from experiment results and agent output files.
 * Run this script to update the dashboard data.
 */

import fs from 'fs';
import path from 'path';
import yaml from 'yaml';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '../..');

const EXPERIMENTS = [
  // Phase 1: Original experiments (C1, Q2 pivoted, Q1 passed)
  { id: 'c1-vlm-latent-sufficiency', name: 'C1: VLM Latent Sufficiency', phase: 1, pivoted: true },
  { id: 'q1-latent-alignment', name: 'Q1: Latent Space Alignment', phase: 1 },
  { id: 'q2-information-preservation', name: 'Q2: Information Preservation', phase: 1, pivoted: true },
  // Phase 1b: Pivot experiment (replaces C1/Q2 for Gate 1)
  { id: 'p2-hybrid-encoder', name: 'P2: Hybrid Encoder (DINOv2 + VLM)', phase: 1 },
  // Phase 2+
  { id: 'c2-adapter-bridging', name: 'C2: Adapter Bridging', phase: 2 },
  { id: 'q3-temporal-coherence', name: 'Q3: Temporal Coherence', phase: 2 },
  { id: 'c3-future-prediction', name: 'C3: Future Prediction', phase: 3 },
  { id: 'q4-training-data', name: 'Q4: Training Data', phase: 3 },
  { id: 'q5-prediction-horizon', name: 'Q5: Prediction Horizon', phase: 3 },
  { id: 'c4-pixel-verification', name: 'C4: Pixel Verification', phase: 4 },
];

const WANDB_BASE = 'https://wandb.ai/a1j9o94/foresight';
const MODAL_BASE = 'https://modal.com/apps/a1j9o94/main';

function readYaml(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return yaml.parse(content);
  } catch (e) {
    return null;
  }
}

function readMarkdown(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch (e) {
    return null;
  }
}

function getExperimentStatus(expId) {
  const resultsPath = path.join(ROOT, 'research/experiments', expId, 'results.yaml');
  const findingsPath = path.join(ROOT, 'research/experiments', expId, 'FINDINGS.md');
  const planPath = path.join(ROOT, 'research/experiments', `${expId}.md`);

  const results = readYaml(resultsPath);
  const findings = readMarkdown(findingsPath);
  const plan = readMarkdown(planPath);

  let status = 'not_started';
  let recommendation = null;
  let metrics = {};
  let subExperiments = [];

  if (results) {
    status = results.status || 'unknown';
    recommendation = results.recommendation;

    if (results.assessment) {
      metrics = {
        lpips: results.assessment.lpips_achieved,
        ssim: results.assessment.ssim_achieved,
        spatial_iou: results.assessment.spatial_iou_achieved,
        confidence: results.assessment.confidence,
      };
    }

    if (results.results?.experiments) {
      subExperiments = Object.entries(results.results.experiments).map(([id, data]) => ({
        id,
        status: data.status,
        finding: data.finding,
        metrics: data.metrics,
        error: data.error || null,
        // failed = code didn't work (has error traceback)
        hasFailed: data.status === 'failed',
      }));
    }
  }

  return {
    status,
    recommendation,
    metrics,
    subExperiments,
    hasFindings: !!findings,
    hasPlan: !!plan,
    findings: findings,
  };
}

function getAgentStatus() {
  const agents = [];

  try {
    // Query Modal for running/recent apps
    const appListOutput = execSync('uv run modal app list --json', {
      cwd: ROOT,
      encoding: 'utf-8',
      timeout: 30000,
      env: { ...process.env, ZDOTDIR: '/dev/null' },
    });

    const apps = JSON.parse(appListOutput);

    // Filter to foresight apps from the last hour
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    const recentApps = apps.filter(app => {
      if (app.Description !== 'foresight') return false;
      const createdAt = new Date(app['Created at']);
      return createdAt > oneHourAgo;
    });

    // For each recent app, try to get the experiment ID from logs
    for (const app of recentApps) {
      const appId = app['App ID'];
      const isRunning = app.State === 'running';

      let experiment = 'Unknown';
      try {
        // Get first few lines of logs to extract experiment ID
        const logsOutput = execSync(`uv run modal app logs ${appId} 2>&1 | head -5`, {
          cwd: ROOT,
          encoding: 'utf-8',
          timeout: 10000,
          env: { ...process.env, ZDOTDIR: '/dev/null' },
          shell: true,
        });

        // Look for "Starting experiment: <experiment-id>" pattern
        const expMatch = logsOutput.match(/Starting experiment:\s*([a-z0-9-]+)/i);
        if (expMatch) {
          experiment = expMatch[1];
        }
      } catch (logErr) {
        // Logs might not be available yet for very new runs
        console.log(`  Could not fetch logs for ${appId}: ${logErr.message}`);
      }

      agents.push({
        id: appId,
        experiment,
        state: app.State,
        createdAt: app['Created at'],
        stoppedAt: app['Stopped at'] || null,
        isActive: isRunning,
        modalUrl: `https://modal.com/apps/a1j9o94/main/deployed/${appId}`,
      });
    }
  } catch (e) {
    console.error('Error querying Modal:', e.message);
  }

  return agents;
}

function generateStatus() {
  const experiments = EXPERIMENTS.map(exp => ({
    ...exp,
    ...getExperimentStatus(exp.id),
  }));

  const agents = getAgentStatus();

  const gates = [
    {
      id: 'gate_1_reconstruction',
      name: 'Gate 1: Reconstruction',
      // Updated after pivot: Q1 passed, P2 replaces failed C1/Q2 spatial requirements
      experiments: ['q1-latent-alignment', 'p2-hybrid-encoder'],
      unlocks: 'Phase 2 (Adapter Training)',
    },
    {
      id: 'gate_2_bridging',
      name: 'Gate 2: Bridging',
      experiments: ['c2-adapter-bridging', 'q3-temporal-coherence'],
      unlocks: 'Phase 3 (Prediction)',
    },
    {
      id: 'gate_3_prediction',
      name: 'Gate 3: Prediction',
      experiments: ['c3-future-prediction', 'q4-training-data', 'q5-prediction-horizon'],
      unlocks: 'Phase 4 (Verification)',
    },
    {
      id: 'gate_4_verification',
      name: 'Gate 4: Verification',
      experiments: ['c4-pixel-verification'],
      unlocks: 'Final Evaluation',
    },
  ];

  // Calculate gate status
  for (const gate of gates) {
    const gateExps = experiments.filter(e => gate.experiments.includes(e.id));
    const passed = gateExps.filter(e => e.status === 'completed' && e.recommendation === 'proceed');
    const pivoted = gateExps.filter(e => e.status === 'completed' && e.recommendation === 'pivot');
    // Failed = code didn't work (threw exceptions)
    const failed = gateExps.filter(e =>
      e.status === 'failed' ||
      e.subExperiments?.some(s => s.hasFailed)
    );

    // Determine gate status
    if (passed.length === gateExps.length) {
      gate.status = 'passed';
    } else if (failed.length > 0) {
      gate.status = 'blocked';  // Blocked by code failures - fix code and re-run
      gate.blockers = failed.map(e => e.id);
      gate.blockerReason = 'failed';
    } else if (pivoted.length > 0) {
      gate.status = 'blocked';  // Blocked by pivot - hypothesis not supported
      gate.blockers = pivoted.map(e => e.id);
      gate.blockerReason = 'pivot';
    } else if (passed.length > 0) {
      gate.status = 'in_progress';
    } else {
      gate.status = 'pending';
    }

    gate.progress = `${passed.length}/${gateExps.length}`;
  }

  const status = {
    generatedAt: new Date().toISOString(),
    experiments,
    agents,
    gates,
    links: {
      wandb: WANDB_BASE,
      modal: MODAL_BASE,
      github: 'https://github.com/a1j9o94/foresight',
    },
  };

  const outputPath = path.join(__dirname, '../public/status.json');
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(status, null, 2));
  console.log(`Status written to ${outputPath}`);

  return status;
}

generateStatus();
