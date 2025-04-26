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

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraRef = useRef(null);
  const wsRef = useRef(null);
  const keypointsBuffer = useRef([]);
  
  const [selectedExerciseId, setSelectedExerciseId] = useState(1);
  const [isExerciseStarted, setIsExerciseStarted] = useState(false);
  const [feedback, setFeedback] = useState("Waiting...");
  const [suggestion, setSuggestion] = useState("");
  const [progressData, setProgressData] = useState({ labels: [], datasets: [] });

  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws/infer");

    wsRef.current.onopen = () => {
      console.log("WebSocket connected");
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setFeedback(data.feedback);
      setSuggestion(data.suggestion);

      if (data.progress) {
        const newLabels = data.progress.timestamps.map((t) => new Date(t).toLocaleTimeString());
        const deviations = data.progress.deviations;

        setProgressData({
          labels: newLabels,
          datasets: [
            {
              label: "Deviation",
              data: deviations,
              backgroundColor: "rgba(75, 192, 192, 0.5)",
              borderColor: "rgba(75, 192, 192, 1)",
              borderWidth: 1,
            },
          ],
        });
      }
    };

    wsRef.current.onclose = () => {
      console.log("WebSocket disconnected");
    };

    return () => {
      wsRef.current?.close();
    };
  }, []);

  const startExercise = async () => {
    if (!selectedExerciseId || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert("WebSocket not connected or Exercise not selected!");
      return;
    }

    setIsExerciseStarted(true);
    setFeedback("Initializing...");
    setSuggestion("");
    setProgressData({ labels: [], datasets: [] });

    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 2,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results) => {
      if (!videoRef.current || !canvasRef.current) return;
      const ctx = canvasRef.current.getContext("2d");

      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
        drawLandmarks(ctx, results.poseLandmarks, { color: "#FF0000", radius: 5 });

        const worldLandmarks = results.poseWorldLandmarks;
        if (worldLandmarks && worldLandmarks.length === 33) {
          const frameKeypoints = worldLandmarks.map(lm => [lm.x, lm.y, lm.z]);
          keypointsBuffer.current.push(frameKeypoints);

          if (keypointsBuffer.current.length >= SEQ_LEN) {
            const payload = {
              label: "keypoint_sequence",
              exercise_id: selectedExerciseId,
              keypoints: keypointsBuffer.current.slice(0, SEQ_LEN),
            };
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify(payload));
            }
            keypointsBuffer.current = []; // clear after sending
          }
        }
      }
    });

    if (videoRef.current) {
      cameraRef.current = new Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current.readyState >= 2) {
            await pose.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
      });
      cameraRef.current.start();
    }
  };

  const stopExercise = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
    }
    keypointsBuffer.current = [];
    setIsExerciseStarted(false);
    setFeedback("Stopped");
    setSuggestion("");
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>AI Exercise Coach</h1>

      <div style={{ marginBottom: "20px" }}>
        <label>
          Select Exercise:
          <select
            value={selectedExerciseId}
            onChange={(e) => setSelectedExerciseId(Number(e.target.value))}
            disabled={isExerciseStarted}
            style={{ marginLeft: "10px", fontSize: "16px" }}
          >
            {Object.entries(EXERCISE_MAP).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
        </label>
        {!isExerciseStarted ? (
          <button
            onClick={startExercise}
            style={{ marginLeft: "20px", padding: "10px 20px", fontSize: "16px", cursor: "pointer" }}
          >
            Start
          </button>
        ) : (
          <button
            onClick={stopExercise}
            style={{ marginLeft: "20px", padding: "10px 20px", fontSize: "16px", backgroundColor: "#f44336", color: "white", cursor: "pointer" }}
          >
            Stop
          </button>
        )}
      </div>

      <div style={{ position: "relative", width: "640px", height: "480px", margin: "auto", border: "1px solid black", overflow: "hidden" }}>
          <video ref={videoRef} width="640" height="480" style={{ position: "absolute", top: 0, left: 0, objectFit: "cover", width: "100%", height: "100%" }} autoPlay muted playsInline />
<         canvas ref={canvasRef} width="640" height="480" style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }} />
      </div>

      <h2 style={{ marginTop: "20px" }}>Feedback: {feedback}</h2>
      {suggestion && <h3>Suggestion: {suggestion}</h3>}

      {progressData.labels.length > 0 && (
        <div style={{ marginTop: "30px", width: "80%", marginLeft: "auto", marginRight: "auto" }}>
          <Bar data={progressData} />
        </div>
      )}
    </div>
  );
}
export default App;
