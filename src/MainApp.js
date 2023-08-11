// App.js
import { Routes, Route } from "react-router-dom";

import Home from "./pages/App";

const App = () => {
	return (
		<>
			<Routes>
				<Route path="/" element={<Home />} />
			</Routes>
		</>
	);
};

export default App;
