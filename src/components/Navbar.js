import React, { useState } from "react";
import CodeIcon from "@mui/icons-material/Code";
import { GiHamburgerMenu } from "react-icons/gi";

import "../App.css";

const Navbar = () => {
	// const [showMediaIcons, setShowMediaIcons] = useState(false);
	return (
		<>
			<nav className="main-nav">
				{/* 1st logo part  */}
				<div className="logo">
					<h2>
						{/* <span>P</span>
						<span>D</span>etection */}
						Plant Disease Detection
					</h2>
				</div>

				{/* 2nd menu part  */}
				{/* <div
					className={
						showMediaIcons ? "menu-link mobile-menu-link" : "menu-link"
					}>
					<ul>
						<li>
							<a to="/">Home</a>
						</li>
						<li>
							<a to="/about">about</a>
						</li>
						<li>
							<a to="/service">services</a>
						</li>
						<li>
							<a to="/contact">Login</a>
						</li>
					</ul>
				</div> */}

				{/* 3rd social media links */}
				{/* <div className="social-media">
					<div className="hamburger-menu">
						<a href="#" onClick={() => setShowMediaIcons(!showMediaIcons)}>
							<GiHamburgerMenu />
						</a>
					</div>
				</div> */}
			</nav>
		</>
	);
};

export default Navbar;
