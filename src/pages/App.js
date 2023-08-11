import React from "react";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Content from "../components/Content";

import "../App.css";

function App() {
	return (
		<div className="App">
			<Navbar />
			<Content />
			<Footer />
		</div>
	);
}

export default App;
