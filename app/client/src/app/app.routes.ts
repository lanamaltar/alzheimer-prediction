import { Routes } from '@angular/router';
import { authGuard } from './auth.guard';
import { LoginComponent } from './login/login.component';
import { RegisterComponent } from './register/register.component';
import { LandingComponent } from './landing/landing.component';
import { ShellComponent } from './shell/shell.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { PredictNumericComponent } from './predict-numeric/predict-numeric.component';
import { NumericPredictionResultComponent } from './numeric-prediction-result/numeric-prediction-result.component';
import { PredictImageComponent } from './predict-image/predict-image.component';
import { HistoryComponent } from './history/history.component';

export const routes: Routes = [
	{ path: '', component: LandingComponent },
	{ path: 'login', component: LoginComponent },
	{ path: 'register', component: RegisterComponent },
	{
		path: '',
		component: ShellComponent,
		children: [
	    { path: 'dashboard', component: DashboardComponent },
		{ path: 'predict/numeric', component: PredictNumericComponent },
	    { path: 'predict/image', component: PredictImageComponent },
	    { path: 'history', component: HistoryComponent },
	  ]
	}
];
