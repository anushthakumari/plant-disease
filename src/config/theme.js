import { createTheme } from "@mui/material/styles";
import { green, orange, purple } from "@mui/material/colors";

const theme = createTheme({
	palette: {
		primary: {
			main: orange[500],
		},
		secondary: {
			main: orange[500],
		},
	},

	typography: {
		fontFamily: ['"Jost"', "sans-serif"].join(","),
		fontSize: 15,
	},
});

export default theme;
