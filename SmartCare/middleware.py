# middleware.py
from django.utils.deprecation import MiddlewareMixin

class RefreshMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        if request.GET.get('refresh') == 'true':
            response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
        return response