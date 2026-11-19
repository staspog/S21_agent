import unittest

from src.session_identity import scoped_thread_id


class SessionIdentityTest(unittest.TestCase):
    def test_session_is_bound_to_verified_subject(self):
        self.assertEqual(scoped_thread_id("a", "session"), scoped_thread_id("a", "session"))
        self.assertNotEqual(scoped_thread_id("a", "session"), scoped_thread_id("b", "session"))
        self.assertNotEqual(scoped_thread_id("a", "bc"), scoped_thread_id("ab", "c"))

    def test_empty_subject_cannot_be_used(self):
        with self.assertRaises(ValueError):
            scoped_thread_id("", "session")
