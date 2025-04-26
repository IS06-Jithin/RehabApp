import { useEffect, useState, useRef } from "react";
import { Pose } from "@mediapipe/pose";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { POSE_CONNECTIONS } from "@mediapipe/pose";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "./App.css";                    // ← ① make sure the CSS is loaded

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const EXERCISE_MAP = {
  1: "Arm abduction",
  2: "Arm VW",
  3: "Push-ups",
  4: "Leg abduction",
  5: "Leg lunge",
  6: "Squats",
};
const SEQ_LEN = 16;

export default function App() {
  const videoRef        = useRef(null);
  const canvasRef       = useRef(null);
  const cameraRef       = useRef(null);
  const wsRef           = useRef(null);
  const keypointsBuffer = useRef([]);
  const [cameraReady, setCameraReady] = useState(false);

  const [selectedExerciseId, setSelectedExerciseId] = useState(1);
  const [isExerciseStarted, setIsExerciseStarted]   = useState(false);
  const [feedback, setFeedback]                     = useState("Waiting…");
  const [suggestion, setSuggestion]                 = useState("");
  const [progressData, setProgressData]             = useState({ labels: [], datasets: [] });
  const hasChart = progressData.labels.length > 0;
  
  /* ─────────────────────────── WebSocket bootstrap ─────────────────────────── */
  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws/infer");

    wsRef.current.onopen = () => console.log("WebSocket connected");

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setFeedback(data.feedback);
      setSuggestion(data.suggestion);

      if (data.progress) {
        const labels = data.progress.timestamps.map((t) =>
          new Date(t).toLocaleTimeString()
        );
        setProgressData({
          labels,
          datasets: [
            {
              label: "Deviation",
              data: data.progress.deviations,
              backgroundColor: "rgba(75, 192, 192, 0.5)",
              borderColor: "rgba(75, 192, 192, 1)",
              borderWidth: 1,
            },
          ],
        });
      }
    };

    wsRef.current.onclose = () => console.log("WebSocket disconnected");
    return () => wsRef.current?.close();
  }, []);

  /* ───────────────────────────── start / stop ──────────────────────────────── */
  const startExercise = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert("WebSocket not connected!"); return;
    }

    setIsExerciseStarted(true);
    setFeedback("Initializing…"); setSuggestion(""); setProgressData({ labels: [], datasets: [] });

    const pose = new Pose({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`,
    });
    pose.setOptions({
      modelComplexity: 2, smoothLandmarks: true,
      minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    });

    pose.onResults((results) => {
      if (!videoRef.current || !canvasRef.current) return;
      if (!cameraReady) setCameraReady(true);
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS,
                       { color: "#00FF00", lineWidth: 4 });
        drawLandmarks(ctx, results.poseLandmarks,
                      { color: "#FF0000", radius: 5 });

        const w = results.poseWorldLandmarks;
        if (w?.length === 33) {
          keypointsBuffer.current.push(w.map((lm) => [lm.x, lm.y, lm.z]));
          if (keypointsBuffer.current.length >= SEQ_LEN) {
            wsRef.current.send(JSON.stringify({
              label: "keypoint_sequence",
              exercise_id: selectedExerciseId,
              keypoints: keypointsBuffer.current.slice(0, SEQ_LEN),
            }));
            keypointsBuffer.current = [];
          }
        }
      }
    });

    if (videoRef.current) {
      cameraRef.current = new Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current.readyState >= 2) await pose.send({ image: videoRef.current });
        },
        width: 640, height: 480,
      });
      cameraRef.current.start();
    }
  };

  const stopExercise = () => {
    cameraRef.current?.stop();
    keypointsBuffer.current = [];
    setIsExerciseStarted(false);
    setFeedback("Stopped"); setSuggestion("");
  };

  /* ────────────────────────────────── UI ───────────────────────────────────── */
  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>RehabApp: AI Physiotherapist</h1>

      {/* selector + start/stop */}
      <div style={{ marginBottom: 20 }}>
      <label>
        Select Exercise:&nbsp;
        <select
          className="exercise-select"          // ← NEW class
          value={selectedExerciseId}
          onChange={(e) => setSelectedExerciseId(+e.target.value)}
          disabled={isExerciseStarted}
        >
          {Object.entries(EXERCISE_MAP).map(([id, name]) => (
            <option key={id} value={id}>{name}</option>
          ))}
        </select>
      </label>

      {isExerciseStarted ? (
        <button
          onClick={stopExercise}
          className="btn btn-stop"             // ← NEW classes
          style={{ marginLeft: 20 }}
        >
          Stop
        </button>
      ) : (
        <button
          onClick={startExercise}
          className="btn btn-start"            // ← NEW classes
          style={{ marginLeft: 20 }}
        >
          Start
        </button>
      )}
    </div>
       {/* ───────── video + chart side-by-side on desktop ───────── */}
       <div className="content-wrapper">
        {/* VIDEO container (video + canvas are always mounted) */}
        <div className="video-container">
          <video
            ref={videoRef}
            width="640"
            height="480"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              objectFit: "cover",
              width: "100%",
              height: "100%",
              visibility: cameraReady ? "visible" : "hidden"
            }}
            autoPlay
            muted
            playsInline
          />
          <canvas
            ref={canvasRef}
            width="640"
            height="480"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              visibility: cameraReady ? "visible" : "hidden"
            }}
          />

          {/* overlay placeholder WHEN camera not ready */}
          {!cameraReady && (
            <div className="video-placeholder">
              <p>
                Click&nbsp;<strong>Start</strong>&nbsp; and wait for the for the video to initilise before begining your therapy
              </p>
            </div>
          )}
        </div>

        {/* CHART / placeholder — unchanged */}
        {progressData.labels.length > 0 ? (
          <div className="chart-container">
            <Bar
              data={progressData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true } },
              }}
            />
          </div>
        ) : (
          isExerciseStarted && (
            <div className="chart-placeholder">
              <p>Your progress will appear here</p>
            </div>
          )
        )}
      </div>
      <h2 style={{ marginTop: 20 }}>Feedback: {feedback}</h2>
      {suggestion && <h3>Suggestion: {suggestion}</h3>}
    </div>
  );
}
