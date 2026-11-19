"""ASGI release identity and drain control; health probes never authenticate upstream."""
import ipaddress
import json
import os

class DeploymentMiddleware:
    def __init__(self, app):
        self.app=app
        self.release_id=os.environ.get('S21_RELEASE_ID','')
        self.active=0
        self.draining=False

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        path=scope.get('path','')
        async def response(status,body):
            data=json.dumps(body).encode()
            await send({'type':'http.response.start','status':status,'headers':[(b'content-type',b'application/json'),(b'x-s21-release',self.release_id.encode())]})
            await send({'type':'http.response.body','body':data})
        if path in ('/health/live','/health/ready'):
            await response(503 if self.draining and path.endswith('ready') else 200,{'status':'not_ready' if self.draining else 'ready','release_id':self.release_id})
            return
        if path.startswith('/_deployment/'):
            try:local=ipaddress.ip_address((scope.get('client') or ('',0))[0]).is_loopback
            except ValueError:local=False
            if not local:return await response(404,{'error':'not_found'})
            if path=='/_deployment/drain' and scope.get('method')=='POST':
                body=b''
                while True:
                    part=await receive();body+=part.get('body',b'')
                    if len(body)>1024:return await response(400,{'error':'invalid_request'})
                    if not part.get('more_body'):break
                try:
                    value=json.loads(body)['draining']
                    if type(value) is not bool:raise ValueError()
                except (ValueError,KeyError,TypeError):return await response(400,{'error':'invalid_request'})
                self.draining=value
            elif path!='/_deployment/status' or scope.get('method')!='GET':return await response(404,{'error':'not_found'})
            return await response(200,{'release_id':self.release_id,'active_requests':self.active,'draining':self.draining})
        if self.draining:return await response(503,{'error':'service_draining'})
        self.active+=1
        async def identified(event):
            if event['type']=='http.response.start':
                event={**event,'headers':[*event.get('headers',[]),(b'x-s21-release',self.release_id.encode())]}
            await send(event)
        try:return await self.app(scope,receive,identified)
        finally:self.active-=1
