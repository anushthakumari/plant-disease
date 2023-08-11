import React, { useEffect, useState } from "react";
import axios from "axios";
import { Typography, CircularProgress } from "@mui/material";
import Button from "@mui/material/Button";

import "../components/styles/Content.css";

const Content = () => {
	const [selectedImage, setselectedImage] = useState();
	const [isLoading, setisLoading] = useState(false);
	const [objurl, setobjurl] = useState("");
	const [result_text, setresult_text] = useState("");

	const onChangeHandler = async (e) => {
		const files = e.target.files;
		if (!files || !files.length) {
			return;
		}
		setresult_text("");
		setisLoading(true);
		setselectedImage(e.target.files[0]);
	};

	const handleUpload = async () => {
		try {
			setisLoading(true);
			setresult_text("");
			const fd = new FormData();
			fd.append("file", selectedImage);
			const { data } = await axios.post("http://localhost", fd);
			setresult_text(data.pred_text);
		} catch (error) {
			console.log(error);
		} finally {
			setisLoading(false);
		}
	};

	useEffect(() => {
		if (selectedImage) {
			setobjurl(URL.createObjectURL(selectedImage));
			setisLoading(false);
		}
	}, [selectedImage]);

	return (
		<div className="mainContainer">
			<img className="plant-img" src="/plantimg.png" alt="plant" />
			<div className="heading">
				<Typography
					variant="h2"
					textAlign={"center"}
					sx={{ padding: "1rem", letterSpacing: ".3rem" }}>
					Detect The Plant Disease
				</Typography>
				<Typography variant="h5">Get a quick check</Typography>
			</div>
			<div className="wrapContainer">
				{/* Code upload */}
				<div className="wholecnt">
					<div className="container">
						<input
							id="file-upload"
							type="file"
							style={{ display: "none" }}
							onChange={onChangeHandler}
							accept=".jpg, jpeg, .png, .webp"
						/>
						{isLoading && (
							<React.Fragment>
								<CircularProgress />
								<Typography variant="h4">Loading...</Typography>
							</React.Fragment>
						)}
						{!selectedImage && !isLoading && (
							<React.Fragment>
								<Button
									variant="contained"
									sx={{
										padding: "15px 30px",
										fontSize: "15px",
										borderRadius: "10px",
										width: 200,
									}}>
									<label htmlFor="file-upload" className="custom-file-upload">
										{"Upload Image"}
									</label>
								</Button>
								<Typography margin={1} fontWeight={600} variant="h6">
									Upload the image of the plant
								</Typography>
							</React.Fragment>
						)}

						{selectedImage && !isLoading && (
							<React.Fragment>
								{result_text ? (
									<React.Fragment>
										<Typography variant="h4">Report</Typography>
										<Typography variant="h5" color={"red"}>
											{result_text}
										</Typography>
									</React.Fragment>
								) : null}
								<Button
									sx={{ fontSize: "1.2rem", textTransform: "capitalize" }}
									variant="link">
									<label htmlFor="file-upload" className="custom-file-upload">
										{"Change Image ?"}
									</label>
								</Button>
								<img src={objurl} alt="uploaded plant" className="previmg" />
								<Button
									variant="contained"
									onClick={handleUpload}
									sx={{
										padding: "10px 30px",
										fontSize: "15px",
										borderRadius: "10px",
										width: 200,
									}}>
									Start Detecting
								</Button>
							</React.Fragment>
						)}
					</div>
				</div>
			</div>
		</div>
	);
};

export default Content;

const readFile = (file) =>
	new Promise(function (resolve, reject) {
		const reader = new FileReader();
		reader.readAsText(file);
		reader.onload = function () {
			const values = reader.result;
			resolve(values);
		};

		reader.onerror = function (e) {
			reject(e);
		};
	});
