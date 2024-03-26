import { useEffect, useState, useRef, useContext } from "react";
import { imgDB, txtDB } from "../../config";
import AuthContext from "../../AuthContext";
import { v4 } from "uuid";
import { getDownloadURL, ref, uploadBytes } from "firebase/storage";
import { addDoc, collection, getDocs } from "firebase/firestore";
import Webcam from "react-webcam"; // Import Webcam component
import "./storeImage.css"; // Import CSS file
import  useSpeechToText  from "../SpeechToText/useSpeechToText"; 
import TextToSpeech from "../TextToSpeech/TextToSpeech";

function StoreImage() {
    const authCtx=useContext(AuthContext);
    const [txt, setTxt] = useState('');
    const [imageUpload,setImageUpload] =useState(null);
    const [img, setImg] = useState('');
    const [imgPreview, setImgPreview] = useState(null); // State to hold image preview
    const [data, setData] = useState([]);
    const [result,setResult] = useState('');
    const [captureMode, setCaptureMode] = useState("webcam"); // State to track capture mode
    const webcamRef = useRef(null); // Reference to the webcam component
    const { isListening, transcript, startListening, stopListening } = useSpeechToText({ continuous: true });


    // Function to capture image from webcam
    const capture = () => {
        const imageSrc = webcamRef.current.getScreenshot();
        console.log(imageSrc)
        setImgPreview(imageSrc);
        // Convert base64 image to Blob
        fetch(imageSrc)
            .then(res => res.blob())
            .then(blob => {
                const file = new File([blob], 'CapturedImage.jpg', { type: 'image/jpeg' });
                setImageUpload(file);
                const imgs= ref(imgDB, `imgs/${authCtx.email}/${img.name + v4()}`);
                uploadBytes(imgs, blob).then(data => {
                    console.log(data, "imgs");
                    getDownloadURL(data.ref).then(val => setImg(val));
                });
            });
    };

   const handleUpload = (e) => {
     setImageUpload(e.target.files[0]);
     console.log(imageUpload);
    const imgs = ref(imgDB, `imgs/${authCtx.email}/${img.name + v4()}`);
    uploadBytes(imgs, e.target.files[0]).then(data => {
        console.log(data, "imgs");
        getDownloadURL(data.ref).then(val => {
            console.log("Download URL:", val);
            setImg(val);
            setImgPreview(val);
            console.log(imgPreview) // Set the image preview URL
        });
    });
   };


    async function getData() {
        const valRef = collection(txtDB, 'txtData');
        const dataDb = await getDocs(valRef);
        const allData = dataDb.docs
        .filter(doc => doc.data().email === authCtx.email) // Filter based on email match
        .map(doc => ({ ...doc.data(), id: doc.id }));
        setData(allData);
        console.log(allData)
    }

    useEffect(() => {
        getData();
    }, []);

    
    const startStopListening =()=>
    {
      isListening? stopVoiceInput():startListening()
    }

    const stopVoiceInput = () => {
        setTxt(prevVal => prevVal + (transcript.length ? (prevVal.length ? ' ' : '') + transcript : ''));
        stopListening();
    };
    
    async function handleClick() {
        setResult("Loading....");
        try {
            // Send the image to the backend for processing
            const formData = new FormData();
            formData.append('image', imageUpload);
            formData.append('text', txt); // Include text input if required by backend
    
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData
            });
    
            if (!response.ok) {
                throw new Error('Failed to send image to backend');
            }
    
            const dat = await response.json();
            console.log('Prediction:', dat.prediction);
            const predicted = dat.prediction[0];
            // Add the result (prediction) along with other data to Firestore
            const valRef = collection(txtDB, 'txtData');
            await addDoc(valRef, { 
                txtval: txt, 
                imgUrl: img,
                email: authCtx.email,
                predictedOutput:predicted
            });
    
            alert("Data added Successfully");
            setResult(predicted);
        } catch (error) {
            console.error('Error handling click:', error);
            alert("Failed to add data. Please try again.");
        }
    };


    return (
        <div className="container">
            {/* Toggle button for capture mode */}
            <div className="outer-container">
            <div className="radio-container">
                <input
                    type="radio"
                    id="webcam"
                    value="webcam"
                    checked={captureMode === "webcam"}
                    onChange={() => setCaptureMode("webcam")}
                />
                <label htmlFor="webcam">Webcam</label>
            </div>
           
            <div className="radio-container">    
                <input
                    type="radio"
                    id="upload"
                    value="upload"
                    checked={captureMode === "upload"}
                    onChange={() => setCaptureMode("upload")}
                />
                <label htmlFor="upload">Upload</label>
            </div>
            </div>
            <div className="capture-container">
            {/* Conditionally render webcam or file input based on capture mode */}
            {captureMode === "webcam" ? (
                <>
                    {/* Webcam component */}
                    <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                    />
                    {/* Capture image button */}
                    <button onClick={capture}>Capture</button>
                    {imgPreview && (
                        <>
                            <h3>Image Preview</h3>
                            <img className="preview-image" src={imgPreview} alt="Preview" />
                        </>
                    )}
                </>
            ) : (
                <>
                    {/* Input for selecting image file */}
                    <label className="custom-file-input">
                      <input type="file" onChange={(e) => handleUpload(e)} />
                        Upload Image
                    </label>
                    {imgPreview && (
                        <>
                            <h3>Image Preview</h3>
                            <img className="preview-image" src={imgPreview} alt="Preview" />
                        </>
                    )}

                </>
            )}
        </div>
        <div className="text-container">
         <div className="text-inner-container">
        <button 
            onClick={startStopListening}
            className={`button ${isListening ? 'isListening' : ''}`}
        >
            {isListening ? 'Stop Listening' : 'Speak'}
        </button>
        <input
            type="text"
            onChange={(e) => setTxt(e.target.value)}
            placeholder="Enter your Question"
             className="text-input"
             disabled={isListening}
                value={isListening ? txt + (transcript.length ? (txt ? ' ' : '') + transcript : '') : txt}
        />
        </div>
            <br />
        <button onClick={handleClick} className="save-button">Submit</button>
        {result !== '' && (
          <div className="result-box">
            <p className="predicted-output">Predicted Output: {result}</p>
                    <TextToSpeech text={result} />
            </div>
        )}
        </div>
        </div>
    );
}

export default StoreImage;
