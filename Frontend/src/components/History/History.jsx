import React, { useState, useEffect, useContext } from 'react';
import { collection, getDocs } from 'firebase/firestore';
import { txtDB } from '../../config';
import AuthContext from "../../AuthContext";
import './History.css'; // Import the CSS file for styling
import Nav from '../Navigation/Nav';

const History = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const authCtx = useContext(AuthContext);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const dataRef = collection(txtDB, 'txtData');
        const querySnapshot = await getDocs(dataRef);

        const retrievedData = [];
        querySnapshot.forEach((doc) => {
          const { txtval, imgUrl, email ,predictedOutput } = doc.data();
          retrievedData.push({ txtval, imgUrl, email, predictedOutput, id: doc.id });
        });

        const userData = retrievedData.filter(item => item.email === authCtx.emailEntered);

        setData(userData);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setLoading(false);
      }
    };

    fetchData();
  }, [authCtx.emailEntered]);

  return (
    <div>
      <Nav/>
      <h2>Previous Uploaded Data</h2>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div className="data-container">
         {data.map((value) => (
             <div key={value.id} className="data-box">
                 <p className="email">Email: {value.email}</p>
                 <p className="text-value">Question: {value.txtval}</p>
                 <img src={value.imgUrl} alt="Image" className="image" />
                 <p className="text-value">Prediction: {value.predictedOutput}</p>
             </div>
         ))}
        </div>
      )}
    </div>
  );
};

export default History;
