import asyncio
import logging
import unittest
from unittest import mock

from src.log_filters import SuccessfulHealthAccessFilter, quiet_health_request, install_health_access_filter
from src.rc.availability import RocketChatAvailability


class HealthLoggingTests(unittest.TestCase):
    def test_health_access_filter_preserves_other_requests_and_failures(self):
        filt = SuccessfulHealthAccessFilter()
        def allowed(method, path, status):
            record = logging.LogRecord('uvicorn.access', logging.INFO, '', 0, '%s %s %s %s %s', ('client', method, path, '1.1', status), None)
            return filt.filter(record)
        for path in ('/health', '/healthz', '/health/live', '/health/ready?check=1'):
            self.assertFalse(allowed('GET', path, 200))
            self.assertTrue(allowed('GET', path, 503))
        self.assertTrue(allowed('POST', '/health', 200))
        self.assertTrue(allowed('GET', '/ask', 200))
        self.assertTrue(quiet_health_request('GET', '/health', 200))
        self.assertFalse(quiet_health_request('GET', '/health', 500))
        malformed = logging.LogRecord('uvicorn.access', logging.INFO, '', 0, 'other', (), None)
        self.assertTrue(filt.filter(malformed))

    def test_install_is_idempotent(self):
        logger = logging.getLogger('uvicorn.access')
        with mock.patch.object(logger, 'filters', []):
            install_health_access_filter()
            install_health_access_filter()
            self.assertEqual(len(logger.filters), 1)


class AvailabilityLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_is_quiet_and_errors_and_recovery_remain(self):
        checker = RocketChatAvailability()
        with mock.patch.object(checker, '_probe', new=mock.AsyncMock(side_effect=[(True, None), (False, 'unavailable'), (True, None)])), mock.patch('src.rc.availability.log') as log:
            for _ in range(3):
                await checker.check(None, force=True)
            log.warning.assert_called_once()
            log.info.assert_called_once_with('RC availability: recovered')
