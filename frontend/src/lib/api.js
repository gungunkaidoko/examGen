/**
 * API client — all calls go through this module so the base URL
 * is defined in one place and can be overridden via env.
 */
import axios from "axios";

// In development, Vite proxies /api/* to http://localhost:8000 (see vite.config.js).
// In production, set VITE_API_URL to your deployed FastAPI base URL.
const BASE_URL = import.meta.env.VITE_API_URL || "";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
});

/** Fetch computed quality metrics for a single exam set. */
export async function fetchMetrics(setNumber) {
  const { data } = await api.get(`/api/exam-sets/${setNumber}/metrics`);
  return data;
}

// ── Exam Sets ──────────────────────────────────────────────────────────────

/** List all generated exam sets with metadata. */
export async function fetchExamSets() {
  const { data } = await api.get("/api/exam-sets");
  return data.exam_sets;
}

/**
 * Fetch a single exam set with optional filters.
 * @param {number} setNumber
 * @param {{ chapter?: string, difficulty?: string, bloom_level?: string, search?: string }} filters
 */
export async function fetchExamSet(setNumber, filters = {}) {
  const params = Object.fromEntries(
    Object.entries(filters).filter(([, v]) => v !== "" && v != null)
  );
  const { data } = await api.get(`/api/exam-sets/${setNumber}`, { params });
  return data;
}

/** Fetch the compact answer key for a set. */
export async function fetchAnswerKey(setNumber) {
  const { data } = await api.get(`/api/exam-sets/${setNumber}/answer-key`);
  return data;
}

/** Return the full PDF download URL for a given set number. */
export function getPdfUrl(setNumber) {
  const origin = BASE_URL || window.location.origin;
  return `${origin}/api/exam-sets/${setNumber}/pdf`;
}

// ── Filters ────────────────────────────────────────────────────────────────

/** Return available values for chapter / difficulty / bloom_level dropdowns. */
export async function fetchFilterOptions() {
  const { data } = await api.get("/api/filters");
  return data;
}

// ── Stats ──────────────────────────────────────────────────────────────────

/** Return aggregate stats across all sets. */
export async function fetchStats() {
  const { data } = await api.get("/api/stats");
  return data;
}

// ── Generation ─────────────────────────────────────────────────────────────

/**
 * Trigger pipeline generation.
 * @param {{ sets: number[], no_db: boolean }} payload
 */
export async function triggerGeneration(payload) {
  const { data } = await api.post("/api/generate", payload);
  return data;
}

/** Poll the status of a generation job. */
export async function fetchGenerationStatus(jobId) {
  const { data } = await api.get("/api/generate/status", {
    params: { job_id: jobId },
  });
  return data;
}

// ── Health ─────────────────────────────────────────────────────────────────

export async function fetchHealth() {
  const { data } = await api.get("/api/health");
  return data;
}
