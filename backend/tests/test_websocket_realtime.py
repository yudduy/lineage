"""
WebSocket Real-time Communication Testing Suite

This test suite covers:
- WebSocket connection establishment and management
- Real-time collaboration features
- Live data updates and synchronization
- Message broadcasting and routing
- Connection resilience and reconnection
- Performance under concurrent connections
- Authentication and authorization over WebSocket
- Event-driven architecture validation
"""

import pytest
import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List
import websockets
from fastapi.testclient import TestClient
from fastapi import WebSocket

from app.main import create_app
from app.services.websocket_manager import WebSocketManager
from app.services.collaboration import CollaborationService
from app.services.auth import create_access_token, verify_token


@pytest.fixture
def app():
    """Create test FastAPI app with WebSocket support."""
    return create_app()


@pytest.fixture
def websocket_manager():
    """Create WebSocket manager for testing."""
    manager = WebSocketManager()
    return manager


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    websocket = MagicMock()
    websocket.send_text = AsyncMock()
    websocket.send_bytes = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.receive_json = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


class TestWebSocketConnection:
    """Test WebSocket connection management."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_establishment(self, websocket_manager, mock_websocket):
        """Test WebSocket connection establishment and registration."""
        user_id = "test_user_123"
        connection_id = str(uuid.uuid4())
        
        # Connect user
        await websocket_manager.connect(mock_websocket, user_id, connection_id)
        
        # Verify connection is registered
        assert user_id in websocket_manager.active_connections
        assert connection_id in websocket_manager.active_connections[user_id]
        assert len(websocket_manager.active_connections[user_id]) == 1
    
    @pytest.mark.asyncio
    async def test_websocket_disconnection(self, websocket_manager, mock_websocket):
        """Test WebSocket disconnection and cleanup."""
        user_id = "test_user_123"
        connection_id = str(uuid.uuid4())
        
        # Connect and then disconnect
        await websocket_manager.connect(mock_websocket, user_id, connection_id)
        await websocket_manager.disconnect(user_id, connection_id)
        
        # Verify connection is cleaned up
        if user_id in websocket_manager.active_connections:
            assert connection_id not in websocket_manager.active_connections[user_id]
    
    @pytest.mark.asyncio
    async def test_multiple_connections_per_user(self, websocket_manager):
        """Test multiple WebSocket connections per user."""
        user_id = "test_user_123"
        
        # Create multiple connections for same user
        connections = []
        for i in range(3):
            mock_ws = MagicMock()
            mock_ws.send_text = AsyncMock()
            connection_id = str(uuid.uuid4())
            
            await websocket_manager.connect(mock_ws, user_id, connection_id)
            connections.append((mock_ws, connection_id))
        
        # Verify all connections are registered
        assert user_id in websocket_manager.active_connections
        assert len(websocket_manager.active_connections[user_id]) == 3
        
        # Test sending message to all user connections
        test_message = {"type": "test", "data": "broadcast test"}
        await websocket_manager.send_to_user(user_id, json.dumps(test_message))
        
        # All connections should receive the message
        for mock_ws, _ in connections:
            mock_ws.send_text.assert_called_with(json.dumps(test_message))
    
    @pytest.mark.asyncio
    async def test_websocket_authentication(self, websocket_manager, mock_websocket):
        """Test WebSocket authentication mechanism."""
        # Create test token
        user_data = {"sub": "test_user_123", "email": "test@example.com"}
        token = create_access_token(data=user_data)
        
        # Test authenticated connection
        connection_id = str(uuid.uuid4())
        
        with patch('app.services.auth.verify_token') as mock_verify:
            mock_verify.return_value = user_data
            
            # Authenticate and connect
            is_authenticated = await websocket_manager.authenticate_connection(token)
            assert is_authenticated is True
            
            await websocket_manager.connect(mock_websocket, user_data["sub"], connection_id)
            
            # Connection should be established
            assert user_data["sub"] in websocket_manager.active_connections
        
        # Test unauthenticated connection
        with patch('app.services.auth.verify_token') as mock_verify:
            mock_verify.side_effect = Exception("Invalid token")
            
            is_authenticated = await websocket_manager.authenticate_connection("invalid_token")
            assert is_authenticated is False


class TestRealTimeCollaboration:
    """Test real-time collaboration features."""
    
    @pytest.mark.asyncio
    async def test_collaboration_session_creation(self, websocket_manager):
        """Test creation and management of collaboration sessions."""
        session_id = str(uuid.uuid4())
        creator_id = "user_creator"
        
        # Create collaboration session
        session_data = {
            "session_id": session_id,
            "name": "Test Collaboration Session",
            "creator_id": creator_id,
            "network_id": "network_123",
            "permissions": "read_write",
            "created_at": datetime.utcnow().isoformat()
        }
        
        await websocket_manager.create_collaboration_session(session_id, session_data)
        
        # Verify session is created
        assert session_id in websocket_manager.collaboration_sessions
        assert websocket_manager.collaboration_sessions[session_id]["creator_id"] == creator_id
    
    @pytest.mark.asyncio
    async def test_user_joining_collaboration_session(self, websocket_manager):
        """Test users joining collaboration sessions."""
        session_id = str(uuid.uuid4())
        creator_id = "user_creator"
        collaborator_id = "user_collaborator"
        
        # Create session
        session_data = {
            "session_id": session_id,
            "name": "Test Session",
            "creator_id": creator_id,
            "participants": []
        }
        await websocket_manager.create_collaboration_session(session_id, session_data)
        
        # Create mock connections
        creator_ws = MagicMock()
        creator_ws.send_text = AsyncMock()
        collaborator_ws = MagicMock()
        collaborator_ws.send_text = AsyncMock()
        
        # Connect users
        await websocket_manager.connect(creator_ws, creator_id, str(uuid.uuid4()))
        await websocket_manager.connect(collaborator_ws, collaborator_id, str(uuid.uuid4()))
        
        # Join session
        await websocket_manager.join_collaboration_session(session_id, collaborator_id)
        
        # Verify user joined session
        session = websocket_manager.collaboration_sessions[session_id]
        assert collaborator_id in session.get("participants", [])
        
        # Creator should be notified of new participant
        creator_ws.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_real_time_collaborative_editing(self, websocket_manager):
        """Test real-time collaborative editing operations."""
        session_id = str(uuid.uuid4())
        user1_id = "user_1"
        user2_id = "user_2"
        
        # Setup collaboration session
        session_data = {
            "session_id": session_id,
            "participants": [user1_id, user2_id],
            "network_state": {
                "nodes": [{"id": "node1", "x": 100, "y": 100}],
                "edges": []
            }
        }
        await websocket_manager.create_collaboration_session(session_id, session_data)
        
        # Create mock connections
        user1_ws = MagicMock()
        user1_ws.send_text = AsyncMock()
        user2_ws = MagicMock()
        user2_ws.send_text = AsyncMock()
        
        await websocket_manager.connect(user1_ws, user1_id, str(uuid.uuid4()))
        await websocket_manager.connect(user2_ws, user2_id, str(uuid.uuid4()))
        
        # User 1 makes an edit
        edit_operation = {
            "type": "node_moved",
            "session_id": session_id,
            "user_id": user1_id,
            "data": {
                "node_id": "node1",
                "new_position": {"x": 150, "y": 120},
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await websocket_manager.broadcast_collaboration_update(session_id, edit_operation, exclude_user=user1_id)
        
        # User 2 should receive the update
        user2_ws.send_text.assert_called()
        
        # Verify the call was made with correct data
        call_args = user2_ws.send_text.call_args[0][0]
        sent_data = json.loads(call_args)
        assert sent_data["type"] == "node_moved"
        assert sent_data["user_id"] == user1_id
    
    @pytest.mark.asyncio
    async def test_conflict_resolution(self, websocket_manager):
        """Test conflict resolution in collaborative editing."""
        session_id = str(uuid.uuid4())
        user1_id = "user_1"
        user2_id = "user_2"
        
        # Setup session with initial state
        session_data = {
            "session_id": session_id,
            "participants": [user1_id, user2_id],
            "network_state": {
                "nodes": [{"id": "node1", "x": 100, "y": 100, "version": 1}],
                "edges": []
            },
            "version": 1
        }
        await websocket_manager.create_collaboration_session(session_id, session_data)
        
        # Simulate conflicting edits
        edit1 = {
            "type": "node_moved",
            "session_id": session_id,
            "user_id": user1_id,
            "data": {
                "node_id": "node1",
                "new_position": {"x": 150, "y": 100},
                "base_version": 1,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        edit2 = {
            "type": "node_moved", 
            "session_id": session_id,
            "user_id": user2_id,
            "data": {
                "node_id": "node1",
                "new_position": {"x": 100, "y": 150},
                "base_version": 1,  # Same base version - conflict!
                "timestamp": (datetime.utcnow() + timedelta(milliseconds=10)).isoformat()
            }
        }
        
        # Apply edits with conflict resolution (last-write-wins or timestamp-based)
        resolved_state = await websocket_manager.resolve_collaboration_conflict(session_id, [edit1, edit2])
        
        # Later timestamp should win
        assert resolved_state["network_state"]["nodes"][0]["y"] == 150
        assert resolved_state["version"] > 1


class TestLiveDataUpdates:
    """Test live data updates and synchronization."""
    
    @pytest.mark.asyncio
    async def test_paper_citation_update_broadcast(self, websocket_manager):
        """Test broadcasting paper citation count updates."""
        # Setup subscribers interested in paper updates
        user_ids = ["user_1", "user_2", "user_3"]
        paper_id = "paper_123"
        
        # Create mock connections
        mock_connections = {}
        for user_id in user_ids:
            mock_ws = MagicMock()
            mock_ws.send_text = AsyncMock()
            mock_connections[user_id] = mock_ws
            await websocket_manager.connect(mock_ws, user_id, str(uuid.uuid4()))
            
            # Subscribe to paper updates
            await websocket_manager.subscribe_to_paper_updates(user_id, paper_id)
        
        # Broadcast citation count update
        update_data = {
            "type": "paper_citation_update",
            "paper_id": paper_id,
            "new_citation_count": 150,
            "previous_count": 125,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        await websocket_manager.broadcast_paper_update(paper_id, update_data)
        
        # All subscribers should receive the update
        for user_id in user_ids:
            mock_connections[user_id].send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_task_progress_updates(self, websocket_manager):
        """Test real-time task progress updates."""
        user_id = "test_user"
        task_id = str(uuid.uuid4())
        
        # Setup connection
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock()
        await websocket_manager.connect(mock_ws, user_id, str(uuid.uuid4()))
        
        # Subscribe to task updates
        await websocket_manager.subscribe_to_task_updates(user_id, task_id)
        
        # Send progress updates
        progress_updates = [
            {"progress": 25, "status": "processing", "message": "Building citation network..."},
            {"progress": 50, "status": "processing", "message": "Analyzing communities..."},
            {"progress": 75, "status": "processing", "message": "Generating visualization..."},
            {"progress": 100, "status": "completed", "message": "Task completed successfully"}
        ]
        
        for update in progress_updates:
            progress_data = {
                "type": "task_progress",
                "task_id": task_id,
                "user_id": user_id,
                **update,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket_manager.send_task_update(user_id, task_id, progress_data)
        
        # Should have sent all progress updates
        assert mock_ws.send_text.call_count == len(progress_updates)
    
    @pytest.mark.asyncio
    async def test_network_visualization_sync(self, websocket_manager):
        """Test real-time network visualization synchronization."""
        session_id = str(uuid.uuid4())
        participants = ["user_1", "user_2", "user_3"]
        
        # Create collaboration session
        session_data = {
            "session_id": session_id,
            "participants": participants,
            "network_state": {
                "nodes": [
                    {"id": "node1", "x": 100, "y": 100},
                    {"id": "node2", "x": 200, "y": 200}
                ],
                "edges": [{"source": "node1", "target": "node2"}]
            }
        }
        await websocket_manager.create_collaboration_session(session_id, session_data)
        
        # Setup connections
        mock_connections = {}
        for user_id in participants:
            mock_ws = MagicMock()
            mock_ws.send_text = AsyncMock()
            mock_connections[user_id] = mock_ws
            await websocket_manager.connect(mock_ws, user_id, str(uuid.uuid4()))
        
        # User 1 updates network layout
        layout_update = {
            "type": "layout_update",
            "session_id": session_id,
            "user_id": "user_1",
            "data": {
                "layout_algorithm": "force_directed",
                "node_positions": {
                    "node1": {"x": 150, "y": 110},
                    "node2": {"x": 250, "y": 210}
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await websocket_manager.broadcast_collaboration_update(
            session_id, layout_update, exclude_user="user_1"
        )
        
        # Other participants should receive layout update
        mock_connections["user_2"].send_text.assert_called()
        mock_connections["user_3"].send_text.assert_called()
        mock_connections["user_1"].send_text.assert_not_called()  # Excluded


class TestWebSocketPerformance:
    """Test WebSocket performance under load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_connections_performance(self, websocket_manager):
        """Test performance with many concurrent connections."""
        num_connections = 100
        connections = []
        connection_times = []
        
        # Create many concurrent connections
        for i in range(num_connections):
            mock_ws = MagicMock()
            mock_ws.send_text = AsyncMock()
            user_id = f"user_{i}"
            connection_id = str(uuid.uuid4())
            
            start_time = time.time()
            await websocket_manager.connect(mock_ws, user_id, connection_id)
            connection_time = time.time() - start_time
            
            connections.append((mock_ws, user_id, connection_id))
            connection_times.append(connection_time)
        
        # All connections should be established quickly
        avg_connection_time = sum(connection_times) / len(connection_times)
        max_connection_time = max(connection_times)
        
        assert avg_connection_time < 0.01  # Average under 10ms
        assert max_connection_time < 0.1   # Max under 100ms
        assert len(websocket_manager.active_connections) == num_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_performance(self, websocket_manager):
        """Test broadcast performance to many connections."""
        num_users = 50
        connections = []
        
        # Setup many connections
        for i in range(num_users):
            mock_ws = MagicMock()
            mock_ws.send_text = AsyncMock()
            user_id = f"user_{i}"
            connection_id = str(uuid.uuid4())
            
            await websocket_manager.connect(mock_ws, user_id, connection_id)
            connections.append((mock_ws, user_id))
        
        # Test broadcast performance
        broadcast_message = {
            "type": "system_announcement",
            "message": "System maintenance in 10 minutes",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        start_time = time.time()
        await websocket_manager.broadcast_to_all(json.dumps(broadcast_message))
        broadcast_time = time.time() - start_time
        
        # Broadcast should be fast
        assert broadcast_time < 1.0  # Should complete within 1 second
        
        # All connections should receive message
        for mock_ws, user_id in connections:
            mock_ws.send_text.assert_called_with(json.dumps(broadcast_message))
    
    @pytest.mark.asyncio
    async def test_message_throughput(self, websocket_manager):
        """Test message throughput under load."""
        user_id = "throughput_test_user"
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock()
        
        await websocket_manager.connect(mock_ws, user_id, str(uuid.uuid4()))
        
        # Send many messages rapidly
        num_messages = 1000
        messages = []
        
        start_time = time.time()
        for i in range(num_messages):
            message = {
                "type": "throughput_test",
                "sequence": i,
                "timestamp": datetime.utcnow().isoformat()
            }
            messages.append(message)
            await websocket_manager.send_to_user(user_id, json.dumps(message))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        messages_per_second = num_messages / total_time
        
        # Should handle high message throughput
        assert messages_per_second > 1000  # At least 1000 messages/second
        assert mock_ws.send_text.call_count == num_messages


class TestWebSocketErrorHandling:
    """Test WebSocket error handling and resilience."""
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, websocket_manager):
        """Test handling of connection failures."""
        user_id = "test_user"
        mock_ws = MagicMock()
        
        # Simulate connection failure during send
        mock_ws.send_text.side_effect = Exception("Connection lost")
        
        await websocket_manager.connect(mock_ws, user_id, str(uuid.uuid4()))
        
        # Sending message should handle failure gracefully
        test_message = {"type": "test", "data": "test message"}
        
        # Should not raise exception
        try:
            await websocket_manager.send_to_user(user_id, json.dumps(test_message))
        except Exception:
            pytest.fail("Should handle connection failure gracefully")
    
    @pytest.mark.asyncio
    async def test_malformed_message_handling(self, websocket_manager):
        """Test handling of malformed messages."""
        user_id = "test_user"
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        
        await websocket_manager.connect(mock_ws, user_id, str(uuid.uuid4()))
        
        # Simulate malformed JSON messages
        malformed_messages = [
            "invalid json",
            '{"incomplete": }',
            '{"missing_type": "data"}',
            "",
            "{}"
        ]
        
        for malformed_msg in malformed_messages:
            mock_ws.receive_text.return_value = malformed_msg
            
            # Should handle malformed messages gracefully
            try:
                await websocket_manager.handle_message(user_id, malformed_msg)
            except Exception as e:
                # Should not propagate exceptions for malformed messages
                assert "malformed" in str(e).lower() or "invalid" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_authentication_failure_handling(self, websocket_manager):
        """Test handling of authentication failures."""
        # Test expired token
        with patch('app.services.auth.verify_token') as mock_verify:
            mock_verify.side_effect = Exception("Token expired")
            
            is_authenticated = await websocket_manager.authenticate_connection("expired_token")
            assert is_authenticated is False
        
        # Test invalid token format
        invalid_tokens = [
            "invalid_token",
            "",
            "Bearer",
            "Bearer ",
            None
        ]
        
        for invalid_token in invalid_tokens:
            is_authenticated = await websocket_manager.authenticate_connection(invalid_token)
            assert is_authenticated is False


# Test fixtures and utilities
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# WebSocket test markers
pytest.mark.websocket = pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")