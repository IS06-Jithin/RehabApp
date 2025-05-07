import React from "react";
import ReactDOM from "react-dom/client";

import { BrowserRouter, Routes, Route } from "react-router-dom";  
import "./index.css";

import App from "./LiveSession";
import FeedbackPage from "./FeedbackPage";                         
import reportWebVitals from "./reportWebVitals";
import SummaryPage from "./SummaryPage";

const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/summary" element={<SummaryPage />} />
        <Route path="/feedback" element={<FeedbackPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);

// optional performance measuring
reportWebVitals();
