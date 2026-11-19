import asyncio
import importlib.util
import json
import pathlib
import unittest
spec=importlib.util.spec_from_file_location('deployment',pathlib.Path(__file__).parents[1]/'src/deployment.py')
deployment=importlib.util.module_from_spec(spec);spec.loader.exec_module(deployment)

class DeploymentTests(unittest.IsolatedAsyncioTestCase):
 async def test_health_does_not_call_application_or_upstream(self):
  async def forbidden(*args):raise AssertionError('health must not call application')
  mw=deployment.DeploymentMiddleware(forbidden)
  async def receive():return {'type':'http.request','body':b''}
  for path in ('/health/live','/health/ready'):
   events=[]
   async def send(e):events.append(e)
   await mw({'type':'http','path':path,'method':'GET','client':('127.0.0.1',12)},receive,send)
   self.assertEqual(events[0]['status'],200)
 async def test_control_is_loopback_only(self):
  mw=deployment.DeploymentMiddleware(None);events=[]
  async def receive():return {'type':'http.request','body':b'{"draining":true}'}
  async def send(e):events.append(e)
  await mw({'type':'http','path':'/_deployment/drain','method':'POST','client':('172.18.0.2',12)},receive,send)
  self.assertEqual(events[0]['status'],404);self.assertFalse(mw.draining)
 async def test_active_request_and_drain(self):
  started=asyncio.Event();finish=asyncio.Event()
  async def app(scope,receive,send):
   started.set();await finish.wait();await send({'type':'http.response.start','status':200,'headers':[]});await send({'type':'http.response.body','body':b'ok'})
  mw=deployment.DeploymentMiddleware(app)
  async def receive():return {'type':'http.request','body':b''}
  async def send(e):pass
  task=asyncio.create_task(mw({'type':'http','path':'/ask','client':('127.0.0.1',12)},receive,send));await started.wait()
  self.assertEqual(mw.active,1);mw.draining=True
  events=[]
  async def collect(e):events.append(e)
  await mw({'type':'http','path':'/ask','client':('127.0.0.1',12)},receive,collect)
  self.assertEqual(events[0]['status'],503)
  finish.set();await task;self.assertEqual(mw.active,0)
