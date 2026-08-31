import unittest
import json
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fleet.core.determinism import DeterministicIdGenerator, SeededRandom
from fleet.core.serialization import canonical_json
from fleet.environments.slack.environment import SlackEnvironment


class SlackExtendedTests(unittest.TestCase):
    def test_deterministic_user_uuids(self) -> None:
        generator = DeterministicIdGenerator(seed=42, namespace="slack")
        uuid1 = generator.uuid("user")
        uuid2 = generator.uuid("user")
        
        self.assertEqual(uuid1, "00000000-0000-0000-0000-000000000001")
        self.assertEqual(uuid2, "00000000-0000-0000-0000-000000000002")
        
        # Test reset restores counter
        generator.reset()
        uuid_reset = generator.uuid("user")
        self.assertEqual(uuid_reset, "00000000-0000-0000-0000-000000000001")

    def test_seeded_random_determinism(self) -> None:
        sr1 = SeededRandom(seed=123, namespace="slack")
        sr2 = SeededRandom(seed=123, namespace="slack")
        
        choices = ["apple", "banana", "cherry", "date"]
        self.assertEqual(sr1.choice(choices), sr2.choice(choices))
        self.assertEqual(sr1.randint(1, 100), sr2.randint(1, 100))

    def test_list_channels(self) -> None:
        environment = SlackEnvironment(seed=1)
        res = environment.execute_tool("list_channels", {}, "U002")
        self.assertFalse(res.error)
        channels = res.payload["output"]["channels"]
        self.assertTrue(len(channels) >= 3)

    def test_create_and_rename_channel(self) -> None:
        environment = SlackEnvironment(seed=1)
        
        # Create Channel
        res = environment.execute_tool("create_channel", {"name": "test-channel", "is_private": True}, "U002")
        self.assertFalse(res.error)
        channel_id = res.payload["output"]["id"]
        # SQLite service IDs are count-based: the seed has C001-C007.
        self.assertEqual("C008", channel_id)
        
        # Check listing shows it
        list_res = environment.execute_tool("list_channels", {}, "U002")
        channel_names = [c["name"] for c in list_res.payload["output"]["channels"]]
        self.assertIn("test-channel", channel_names)
        
        # Rename channel
        rename_res = environment.execute_tool("change_channel_name", {"channel_id": channel_id, "new_name": "renamed-channel"}, "U002")
        self.assertFalse(rename_res.error)
        self.assertEqual(rename_res.payload["output"]["channel"]["name"], "renamed-channel")

    def test_update_message(self) -> None:
        environment = SlackEnvironment(seed=1)
        # Post a message
        post_res = environment.execute_tool("post_message", {"channel_id": "C001", "body": "Original body"}, "U002")
        self.assertFalse(post_res.error)
        message_id = post_res.payload["output"]["id"]
        
        # Update message by author
        update_res = environment.execute_tool("update_message", {"message_id": message_id, "body": "Updated body"}, "U002")
        self.assertFalse(update_res.error)
        self.assertEqual(update_res.payload["output"]["message"]["body"], "Updated body")
        self.assertIsNotNone(update_res.payload["output"]["message"]["edited_at_ms"])
        
        # Unauthorized update should fail
        fail_res = environment.execute_tool("update_message", {"message_id": message_id, "body": "Hacked body"}, "U003")
        self.assertTrue(fail_res.error)
        self.assertEqual(fail_res.error.error_code, "permission_denied")

    def test_group_chats(self) -> None:
        environment = SlackEnvironment(seed=1)
        
        # Create group chat
        create_res = environment.execute_tool("create_group", {"name": "SRE Sync", "participants": ["U001", "U003"]}, "U002")
        self.assertFalse(create_res.error)
        group_id = create_res.payload["output"]["id"]
        # SQLite service IDs are count-based: the seed has chats G001 and D001.
        self.assertEqual("G003", group_id)
        
        # Verify participants contain full User models
        participants = create_res.payload["output"]["chat"]["participants"]
        self.assertEqual(len(participants), 3) # U002 (actor), U001, U003
        self.assertTrue(all("display_name" in p for p in participants))
        
        # Send group message
        msg_res = environment.execute_tool("send_group_message", {"group_id": group_id, "body": "Hello SREs!"}, "U002")
        self.assertFalse(msg_res.error)
        self.assertEqual(msg_res.payload["output"]["message"]["channel_id"], group_id)
        
        # Rename group
        rename_res = environment.execute_tool("change_group_name", {"group_id": group_id, "new_name": "SRE & Platform Sync"}, "U002")
        self.assertFalse(rename_res.error)
        self.assertEqual(rename_res.payload["output"]["chat"]["name"], "SRE & Platform Sync")

    def test_dm_chats(self) -> None:
        environment = SlackEnvironment(seed=1)
        
        # Create DM chat container
        create_res = environment.execute_tool("create_dm_message", {"recipient_id": "U003"}, "U002")
        self.assertFalse(create_res.error)
        dm_chat_id = create_res.payload["output"]["id"]
        # SQLite service IDs are count-based: the seed has chats G001 and D001.
        self.assertEqual("D003", dm_chat_id)
        
        # Send DM using chat_id
        send1 = environment.execute_tool("send_dm_message", {"chat_id": dm_chat_id, "body": "Hey Cara"}, "U002")
        self.assertFalse(send1.error)
        
        # Send DM using recipient_id (should auto-locate same DM chat)
        send2 = environment.execute_tool("send_dm_message", {"recipient_id": "U003", "body": "How is it going?"}, "U002")
        self.assertFalse(send2.error)
        self.assertEqual(send2.payload["output"]["message"]["channel_id"], dm_chat_id)

    def test_change_display_name(self) -> None:
        environment = SlackEnvironment(seed=1)
        
        # Change own display name
        res = environment.execute_tool("change_user_display_name", {"user_id": "U002", "new_display_name": "Benjamin Ortiz"}, "U002")
        self.assertFalse(res.error)
        self.assertEqual(res.payload["output"]["user"]["display_name"], "Benjamin Ortiz")
        
        # Change other user's display name without admin role should fail
        fail_res = environment.execute_tool("change_user_display_name", {"user_id": "U001", "new_display_name": "Hacked Alice"}, "U002")
        self.assertTrue(fail_res.error)
        
        # Admin changing other user's display name should pass
        admin_res = environment.execute_tool("change_user_display_name", {"user_id": "U002", "new_display_name": "Ben O."}, "U001")
        self.assertFalse(admin_res.error)

    def test_privacy_aware_search(self) -> None:
        environment = SlackEnvironment(seed=1)
        
        # Create private SRE group
        group_res = environment.execute_tool("create_group", {"name": "Secret Group", "participants": ["U001"]}, "U002")
        group_id = group_res.payload["output"]["id"]
        
        # Post private secret message
        environment.execute_tool("send_group_message", {"group_id": group_id, "body": "super secret key is 1234"}, "U002")
        
        # Actor U002 can search it
        search1 = environment.execute_tool("search_messages", {"query": "secret"}, "U002")
        self.assertEqual(search1.payload["output"]["count"], 1)
        
        # Non-member U003 cannot search/see it
        search2 = environment.execute_tool("search_messages", {"query": "secret"}, "U003")
        self.assertEqual(search2.payload["output"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
