import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";

import "./App.css";

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Main from "./Main";

function App() {
  return (
    <div>
      <Router>
        <Routes>
          <Route path="/" element={<Main />} />
        </Routes>
      </Router>
    </div>
  );
}
export default App;
