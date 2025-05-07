/* === SummaryPage.jsx === */
import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Pie, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
} from "chart.js";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement
);

const EXERCISE_MAP = {
  1: "Arm abduction",
  2: "Arm VW",
  3: "Push-ups",
  4: "Leg abduction",
  5: "Leg lunge",
  6: "Squats",
};

export default function SummaryPage() {
  const { state } = useLocation();
  const navigate = useNavigate();

  const kpi = state?.kpi ?? { avg: "0.0", correct: 0, total: 0 };
  const jointMean = state?.jointMean ?? Array(14).fill(0);
  const lastExId = state?.selectedExercise ?? 1;
  const focusMsg  = state?.focusMsg ?? "";

  /* --- chart datasets --- */
  const pieData = {
    labels: ["Correct", "Incorrect"],
    datasets: [
      {
        data: [kpi.correct, Math.max(kpi.total - kpi.correct, 0)],
        backgroundColor: ["#4caf50", "#f44336"],
        borderWidth: 1,
      },
    ],
  };

  const barData = {
    labels: [
      "L-elbow",
      "R-elbow",
      "L-shoulder",
      "R-shoulder",
      "L-hip",
      "R-hip",
      "L-knee",
      "R-knee",
      "Spine",
      "Head",
      "L-wrist",
      "R-wrist",
      "L-ankle",
      "R-ankle",
    ],
    datasets: [
      {
        data: jointMean,
        backgroundColor: "#2196f3",
        borderWidth: 1,
      },
    ],
  };

  return (
    <div style={{ textAlign: "center", padding: 20 }}>
      <h1>Session summary</h1>
      <h2>{EXERCISE_MAP[lastExId]}</h2>

      {/* KPI cards */}
      <div className="kpi-grid" style={{ marginTop: 10 }}>
        <div className="kpi-card">
          <h4>Avg error</h4>
          <span>{kpi.avg}°</span>
        </div>
        <div className="kpi-card">
          <h4>% correct</h4>
          <span>{kpi.total ? Math.round((100 * kpi.correct) / kpi.total) : 0}%</span>
        </div>
      </div>

      {/* pie */}
      <div style={{ width: 220, margin: "20px auto" }}>
        <Pie
          data={pieData}
          options={{
            plugins: { legend: { display: true, position: "bottom" } },
            maintainAspectRatio: false,
          }}
        />
      </div>

      {/* bar */}
      <div style={{ width: 420, height: 260, margin: "20px auto" }}>
        <Bar
          data={barData}
          options={{
            plugins: { legend: { display: false } },
            scales: {
              y: {
                beginAtZero: true,
                title: { text: "° deviation", display: true },
              },
            },
            maintainAspectRatio: false,
          }}
        />
      </div>
      {/* top-joint advice */}
      {focusMsg && (
        <p style={{ fontWeight: 600, marginTop: 10 }}>{focusMsg}</p>
      )}

      {/* start over */}
      <button
        className="btn btn-start"
        style={{ marginTop: 10 }}
        onClick={() => navigate("/", { state: { selectedExercise: lastExId } })}
      >
        Start over
      </button>

      {/* feedback button */}
      <div style={{ marginTop: 30 }}>
        <button
          className="btn btn-feedback"
          onClick={() => navigate("/feedback")}
        >
          Your Feedbacks
        </button>
      </div>
    </div>
  );
}
