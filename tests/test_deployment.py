"""
tests/test_deployment.py — Unit tests for Phase 6 Deployment features.
Tests container health probes, readiness probes, security headers, and config defaults.
"""

import unittest

# Import app modules
from core.config import AppConfig
from app import flask_app


class TestDeploymentFeatures(unittest.TestCase):
    def setUp(self):
        self.client = flask_app.test_client()
        self.client.testing = True

    def test_health_probe(self):
        """Test GET /health returns HTTP 200 and valid JSON schema."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertIn('uptime_seconds', data)
        self.assertIn('timestamp', data)
        self.assertEqual(data.get('version'), '6.0.0')

    def test_ready_probe(self):
        """Test GET /ready returns HTTP 200 or 503 with structured checks."""
        response = self.client.get('/ready')
        self.assertIn(response.status_code, (200, 503))
        data = response.get_json()
        self.assertIn('checks', data)
        self.assertIn('storage_writable', data['checks'])

    def test_security_headers(self):
        """Test HTTP response includes standard security headers."""
        response = self.client.get('/health')
        headers = response.headers
        self.assertEqual(headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(headers.get('X-Frame-Options'), 'SAMEORIGIN')
        self.assertEqual(headers.get('X-XSS-Protection'), '1; mode=block')
        self.assertEqual(headers.get('Referrer-Policy'), 'strict-origin-when-cross-origin')
        self.assertIn('Content-Security-Policy', headers)

    def test_config_from_env(self):
        """Test AppConfig parses environment variables properly."""
        cfg = AppConfig.from_env()
        self.assertIsInstance(cfg.port, int)
        self.assertIsInstance(cfg.debug, bool)
        self.assertIsInstance(cfg.security_headers_enabled, bool)

    def test_root_serves_index_html(self):
        """Test GET / returns HTTP 200 and serves index.html."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'StockBuddy', response.data)


if __name__ == '__main__':
    unittest.main()
