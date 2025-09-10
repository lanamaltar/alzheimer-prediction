import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-landing',
  standalone: true,
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.component.scss']
})
export class LandingComponent {
  constructor(private router: Router) {}

  openLogin() {
    this.router.navigate(['/login']);
  }

  openRegister() {
    this.router.navigate(['/register']);
  }
}
