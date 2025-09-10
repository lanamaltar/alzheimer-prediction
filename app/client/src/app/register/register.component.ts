import { Component } from '@angular/core';
import { ApiService } from '../api.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.scss']
})
export class RegisterComponent {
  username = '';
  password = '';
  message = '';
  success = false;

  constructor(private apiService: ApiService, private router: Router) {}

  onRegister() {
    this.apiService.register({ username: this.username, password: this.password }).subscribe({
      next: (res) => {
        this.success = res.success;
        this.message = res.message || 'Registration successful!';
        if (res.success) {
          // Automatically log in after successful registration
          this.apiService.login({ username: this.username, password: this.password }).subscribe({
            next: (loginRes) => {
              if (loginRes.success && loginRes.token) {
                localStorage.setItem('jwt_token', loginRes.token);
                localStorage.setItem('userName', this.username);
                this.message = 'Registration and login successful!';
                this.username = '';
                this.password = '';
                // Redirect to dashboard after registration+login
                this.router.navigate(['/dashboard']);
              } else {
                this.message = 'Registration successful, but login failed.';
              }
            },
            error: () => {
              this.message = 'Registration successful, but login failed.';
            }
          });
        }
      },
      error: (err) => {
        this.success = false;
        this.message = err.error?.message || 'Registration failed. Please try again.';
      }
    });
  }
}
