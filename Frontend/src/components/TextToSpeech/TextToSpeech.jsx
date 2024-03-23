import React, { useState, useEffect } from 'react';
import "./TextToSpeech.css";

const TextToSpeech = ({ text }) => {
    const [voices, setVoices] = useState([]);
    const [selectedVoice, setSelectedVoice] = useState(null);

    useEffect(() => {
        const speechVoices = window.speechSynthesis.getVoices();
        setVoices(speechVoices);
        const defaultVoice = speechVoices.find(voice => voice.default);
        setSelectedVoice(defaultVoice || speechVoices[0]);

        const handleVoicesChanged = () => {
            const updatedVoices = window.speechSynthesis.getVoices();
            setVoices(updatedVoices);
            const defaultVoice = updatedVoices.find(voice => voice.default);
            setSelectedVoice(defaultVoice || updatedVoices[0]);
        };

        window.speechSynthesis.onvoiceschanged = handleVoicesChanged;

        return () => {
            window.speechSynthesis.onvoiceschanged = null;
        };
    }, []);

    const handleChangeVoice = (event) => {
        const selectedVoiceIndex = parseInt(event.target.value);
        setSelectedVoice(voices[selectedVoiceIndex]);
    };

    const handleSpeak = () => {
        const speech = new SpeechSynthesisUtterance();
        speech.text = text;
        if (selectedVoice) {
            speech.voice = selectedVoice;
        }
        window.speechSynthesis.speak(speech);
    };

    return (
        <div className="text-to-speech-container">
            <select onChange={handleChangeVoice} className="voice-select">
                {voices.map((voice, index) => (
                    <option key={index} value={index}>
                        {voice.name}
                    </option>
                ))}
            </select>
            <button onClick={handleSpeak} className="speak-button">Listen</button>
        </div>
    );
};

export default TextToSpeech;
