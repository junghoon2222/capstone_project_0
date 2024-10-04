import "react-clock/dist/Clock.css";
import "./Clock.css";
import { useEffect, useState } from "react";
import moment from "moment";
import "moment/locale/ko";

function Clock() {
  const [value, setValue] = useState(moment());

  useEffect(() => {
    const interval = setInterval(() => setValue(moment()), 1000);
    return () => {
      clearInterval(interval);
    };
  }, []);
  const ampm = value.locale("ko").format("A"); // 오전/오후
  const time = value.format("HH:mm:ss"); // 시간

  return (
    <div className="mb-0" style={{ marginTop: "80px", marginBottom: "0px" }}>
      <div className="time">
        <span className="ampm">{ampm}</span> {/* 오전/오후 */}
        <span>{time}</span> {/* 시간 */}
      </div>
      <div style={{ marginTop: "-8px" }}>
        <div className="date">
          <span style={{ fontWeight: "bold" }}>
            {value.locale("ko").format("(ddd) ")}
          </span>
          {value.locale("ko").format("MMMM Do, YYYY")}
        </div>
      </div>
    </div>
  );
}

export default Clock;
