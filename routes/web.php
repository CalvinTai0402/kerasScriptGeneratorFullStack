<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\KerasModelController;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

Route::get('/', function () {
    return view('reactApp');
});
Route::resource("kerasModel", KerasModelController::class);
Route::post("kerasModel/deleteMany",  [KerasModelController::class, 'destroyMany'])->name('kerasModels.destroyMany');
Route::get("kerasModel/{kerasModel}/downloadFile",  [KerasModelController::class, 'downloadFile'])->name('kerasModels.downloadFile');
