import React, { useState, useEffect } from "react";
import "./Cards.css";

function Card() {
  const toggleExpand = (index) => {
    setCards((prevCards) =>
      prevCards.map((card, i) =>
        i === index ? { ...card, expanded: !card.expanded } : card
      )
    );
  };
  const [cards, setCards] = useState([
    {
      title: "",
      items: [""],
      expanded: true,
    },
  ]);
  return (
    <div className="d-flex flex-wrap flex-column me-2 text-center">
      {cards.map((card, index) => (
        <div key={index} className="ms-5 me-2 mb-3 pb-4 text-start">
          <div className="p-1">
            <div
              className="ms-4 mt-0 ps-3 text-center"
              style={{ color: "#FFA0CB", fontSize: "27px" }}
            >
              {card.title}
            </div>
            {card.expanded ? (
              <div className="ps-5">
                {card.items.map((item, itemIndex) => (
                  <div key={itemIndex}>
                    <h6
                      className="fw-light mt-3 ms-0 mb-0 text-start text-gray"
                      style={{
                        fontSize: "25px",
                        fontFamily: "GmarketSansLIght",
                      }}
                    >
                      {item}
                    </h6>
                  </div>
                ))}
              </div>
            ) : (
              <div className="mt-1 ms-1 mb-0 me-4 fs-5 text-white-50"> </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default Card;
