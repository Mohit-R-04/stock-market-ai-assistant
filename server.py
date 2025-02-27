import asyncio
import websockets
import json
from datetime import datetime
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import os
from aiohttp import web
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download warning: {e}")

class ChatbotServer:
    def __init__(self, host='0.0.0.0', port=None):
        self.host = host
        self.port = int(os.environ.get('PORT', 12346))
        self.clients = {}
        
        # Load knowledge base and initialize models
        try:
            self.knowledge_base = pd.read_csv('knowledge_base.csv', quoting=1, encoding='utf-8')
            if 'question' not in self.knowledge_base.columns or 'answer' not in self.knowledge_base.columns:
                raise ValueError("CSV file must contain 'question' and 'answer' columns")
            logger.info(f"Loaded {len(self.knowledge_base)} FAQ entries")
            
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.question_embeddings = self.model.encode(self.knowledge_base['question'].tolist())
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
            logger.info("NLP models loaded successfully")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def preprocess_text(self, text):
        """Preprocess text for better matching"""
        try:
            text = re.sub(r'[^\w\s]', '', text.lower())
            tokens = nltk.word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text

    def get_stock_price(self, ticker):
        """Fetch real-time stock price using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                price = data['Close'].iloc[-1]
                return f"The current price of {ticker.upper()} is ${price:.2f}."
            return f"Could not fetch price for {ticker.upper()}."
        except Exception as e:
            logger.error(f"Error fetching stock price for {ticker}: {e}")
            return "Sorry, I couldn't retrieve the stock price at this time."

    async def find_best_answer(self, question):
        """Enhanced answer matching with real-time stock data"""
        try:
            # Check for stock price requests
            price_match = re.search(r'(\w+)\s*(price|stock price)', question, re.IGNORECASE)
            if price_match:
                ticker = price_match.group(1)
                return {
                    'answer': self.get_stock_price(ticker),
                    'confidence': 1.0,
                    'similar_questions': []
                }

            # FAQ matching
            processed_question = self.preprocess_text(question)
            question_embedding = self.model.encode([processed_question])
            similarities = cosine_similarity(question_embedding, self.question_embeddings)[0]
            
            top_indices = similarities.argsort()[-5:][::-1]
            top_scores = similarities[top_indices]
            
            if top_scores[0] > 0.8:
                return {
                    'answer': self.knowledge_base.iloc[top_indices[0]]['answer'],
                    'confidence': float(top_scores[0]),
                    'similar_questions': self.knowledge_base.iloc[top_indices[1:]]['question'].tolist()
                }
            elif top_scores[0] > 0.5:
                return {
                    'answer': f"Here's what I found that might help: \n\n{self.knowledge_base.iloc[top_indices[0]]['answer']}",
                    'confidence': float(top_scores[0]),
                    'similar_questions': self.knowledge_base.iloc[top_indices[1:]]['question'].tolist(),
                    'suggestion': "You might also be interested in these related topics:"
                }
            else:
                related_questions = self.knowledge_base.iloc[top_indices]['question'].tolist()
                return {
                    'answer': "I'm not sure about that specific question, but I can help with these related topics:",
                    'confidence': 0.0,
                    'similar_questions': related_questions
                }
        except Exception as e:
            logger.error(f"Error finding answer: {e}")
            return {
                'answer': "I apologize, but I encountered an error processing your question.",
                'confidence': 0.0
            }

    async def handle_message(self, websocket, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            question = data.get('content', '').strip()
            
            if question:
                response = await self.find_best_answer(question)
                await websocket.send(json.dumps({
                    'type': 'message',
                    'sender': 'AI Assistant',
                    'content': response['answer'],
                    'confidence': response['confidence'],
                    'similar_questions': response.get('similar_questions', []),
                    'suggestion': response.get('suggestion', ''),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }))
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            await websocket.send(json.dumps({'type': 'error', 'content': 'Invalid message format'}))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await websocket.send(json.dumps({'type': 'error', 'content': 'An error occurred'}))

    async def handler(self, websocket):
        """Main handler for each client connection"""
        try:
            self.clients[websocket] = {"joined_at": datetime.now()}
            logger.info(f"New client connected. Total clients: {len(self.clients)}")
            
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        finally:
            if websocket in self.clients:
                del self.clients[websocket]
                logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def start(self):
        """Start the combined HTTP and WebSocket server"""
        app = web.Application()
        app.router.add_get('/', self.handle_http)
        app['websockets'] = websockets.serve(self.handler, self.host, self.port)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Server started on http://{self.host}:{self.port} and ws://{self.host}:{self.port}")
        await asyncio.Future()

    async def handle_http(self, request):
        """Serve the index.html file"""
        try:
            with open('index.html', 'r', encoding='utf-8') as f:
                return web.Response(text=f.read(), content_type='text/html')
        except FileNotFoundError:
            return web.Response(text="Page not found", status=404)

def main():
    try:
        server = ChatbotServer()
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()