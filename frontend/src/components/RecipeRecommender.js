import React, { useState, useRef, useEffect } from "react";

// replace with urs
const API_BASE_URL = "";
const LOCAL_DJANGO_URL = "http://127.0.0.1:8000/api/generate_recipe/";
const IMAGE_UPLOAD_URL = "http://127.0.0.1:8000/api/detect_ingredients/";

const RecipeRecommender = () => {
    const [userId, setUserId] = useState("");
    const [userPrompt, setUserPrompt] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [selectedImage, setSelectedImage] = useState(null);
    const chatEndRef = useRef(null);

    const formatRecipe = (text) => {
        if (!text) return "";
        return text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br />");
    };

    const fetchChatHistory = async () => {
        if (!userId) return;
        try {
            const response = await fetch(API_BASE_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "get_chat_history", user_id: userId }),
            });

            if (!response.ok) throw new Error(`http error: ${response.status}`);

            const data = await response.json();
            setChatHistory(data.map(chat => ({
                prompt: chat.prompt,
                recipe: formatRecipe(chat.recipe)
            })));
        } catch (error) {
            console.error("error fetching chat history:", error);
        }
    };

    const handleLogin = async () => {
        if (userId.trim() === "") {
            alert("please enter a valid user id.");
            return;
        }
        setIsLoggedIn(true);
        await fetchChatHistory();
    };

    const fetchRecipe = async (providedUserId, providedPrompt) => {
        const finalUserId = providedUserId || userId;
        const finalPrompt = providedPrompt || userPrompt;
    
        if (!finalUserId || !finalPrompt.trim()) {
            console.error("error: missing user_id or prompt");
            return;
        }
    
        const requestData = { user_id: finalUserId, prompt: finalPrompt };
    
        try {
            const response = await fetch(LOCAL_DJANGO_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestData),
            });
    
            const text = await response.text();
            if (!response.ok) {
                console.error("http error:", response.status);
                return;
            }
    
            const data = JSON.parse(text);
            console.log("recipe received:", data);
    
            const newChat = { prompt: finalPrompt, recipe: formatRecipe(data.recipe || "no recipe available.") };
            setChatHistory(prevHistory => [...prevHistory, newChat]);
            await storeChatHistory(newChat);
            setUserPrompt("");
        } catch (error) {
            console.error("error fetching recipe:", error);
        }
    };
    
    const storeChatHistory = async (newChat) => {
        try {
            await fetch(API_BASE_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "store_chat_history", user_id: userId, chat_entry: newChat }),
            });
        } catch (error) {
            console.error("error storing chat history:", error);
        }
    };

    const handleImageUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        setSelectedImage(file);
    
        const formData = new FormData();
        formData.append("user_id", userId);
        formData.append("image", file);
    
        try {
            const response = await fetch(IMAGE_UPLOAD_URL, {
                method: "POST",
                body: formData,
            });
    
            if (!response.ok) throw new Error(`image upload failed: ${response.status}`);
    
            const data = await response.json();
            console.log("detected ingredients:", data.ingredients);
    
            if (data.ingredients.length > 0) {
                const generatedPrompt = `Create a recipe using ${data.ingredients.join(", ")}`;
                setUserPrompt(generatedPrompt);
                await fetchRecipe(userId, generatedPrompt);
            }
        } catch (error) {
            console.error("error detecting ingredients:", error);
        }
    };
    
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatHistory]);

// frontend

    return (
        <div style={{ padding: "20px", textAlign: "center" }}>
            {!isLoggedIn ? (
                <div>
                    <h2>enter user id</h2>
                    <input
                        type="text"
                        placeholder="enter your user id"
                        value={userId}
                        onChange={(e) => setUserId(e.target.value)}
                    />
                    <button onClick={handleLogin}>continue</button>
                </div>
            ) : (
                <div>
                    <h2>recipe recommender</h2>

                    <div style={{
                        border: "1px solid #ccc",
                        padding: "10px",
                        maxHeight: "700px",
                        overflowY: "auto",
                        textAlign: "left",
                        marginBottom: "15px"
                    }}>
                        {chatHistory.map((chat, index) => (
                            <div key={index} style={{ marginBottom: "10px" }}>
                                <strong>user:</strong> {chat.prompt} <br />
                                <strong>recipe:</strong>
                                <div dangerouslySetInnerHTML={{ __html: chat.recipe }} />
                            </div>
                        ))}
                        <div ref={chatEndRef} />
                    </div>

                    <input
                        type="text"
                        placeholder="enter your recipe request"
                        value={userPrompt}
                        onChange={(e) => setUserPrompt(e.target.value)}
                    />
                    <button onClick={fetchRecipe}>generate recipe</button>

                    <div style={{ marginTop: "15px" }}>
                        <input type="file" accept="image/*" onChange={handleImageUpload} />
                        {selectedImage && <p>image selected: {selectedImage.name}</p>}
                    </div>
                </div>
            )}
        </div>
    );
};

export default RecipeRecommender;
