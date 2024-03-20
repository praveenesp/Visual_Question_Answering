import "./History.css";
export default function History({ value }) {
    return (
        <div className="history-container">
            <img className="history-image" src={value.imgUrl} alt="Uploaded" height="200px" width="200px" />
            <h1 className="history-text">{value.txtval}</h1>
        </div>
    );
}
