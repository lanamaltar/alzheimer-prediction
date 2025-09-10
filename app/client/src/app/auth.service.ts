import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  logout() {
    // Clear user authentication data (e.g., tokens, session storage)
  localStorage.removeItem('jwt_token');
  localStorage.removeItem('userName');
  }

  isAuthenticated(): boolean {
    // Check if the user is authenticated (e.g., token exists)
  return !!localStorage.getItem('jwt_token');
  }

  getUserName(): string {
    // Fetch the user's name (replace with actual logic, e.g., from a token or API)
    return localStorage.getItem('userName') || 'Guest';
  }
}
