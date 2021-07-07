<?php

namespace App\Http\Controllers;

use App\Models\KerasModel;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Storage;
use Config;

class KerasModelController extends Controller
{
    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function index(Request $request)
    {
        $search = $request->input("search");
        $limit = $request->input("limit");
        $page = $request->input("page");
        $orderBy = $request->input("orderBy");
        $order = $request->input("order");
        $toSkip = ($page - 1) * $limit;
        $kerasModels = KerasModel::fileName($search)
            ->description($search)
            ->order($orderBy, $order)
            ->skipPage($toSkip)
            ->take($limit)
            ->get();
        return response()->json(['count' => KerasModel::count(), 'total' => KerasModel::count(), 'data' => $kerasModels]);
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\Response
     */
    public function store(Request $request)
    {
        $response = [];
        $validator = Validator::make(
            $request->all(),
            [
                'kerasModelFile' => 'required|file|mimes:zip|max:204800',
                'kerasModelFile.*' => 'required|file|mimes:zip|max:204800'
            ]
        );
        if ($validator->fails()) {
            return response()->json(["status" => "failed", "message" => "Validation error", "errors" => $validator->errors()]);
        }
        if ($request->has('kerasModelFile')) {
            $kerasModelFile = $request->file('kerasModelFile');
            $filename = date("Ymd_His", time()) . $kerasModelFile->getClientOriginalName();
            if (config('env.appEnv') == "local") {
                $kerasModelFile->move('kerasModels/', $filename);
            } elseif (config('env.appEnv') == "production") {
                $kerasModelFile->storeAs('kerasModels/', $filename, 's3');
            }
            KerasModel::create([
                'file_name' => $request->input("file_name"),
                'description' => $request->input("description"),
                'kerasModelFile' => $filename,
            ]);
            $response["status"] = 201;
            $response["message"] = "Success! kerasModelFile(s) uploaded";
        } else {
            $response["status"] = 500;
            $response["message"] = "Failed! kerasModelFile(s) not uploaded";
        }
        return response()->json($response);
    }

    /**
     * Display the specified resource.
     *
     * @param  \App\Models\KerasModel  $kerasModel
     * @return \Illuminate\Http\Response
     */
    public function show(KerasModel $kerasModel)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     *
     * @param  \App\Models\KerasModel  $kerasModel
     * @return \Illuminate\Http\Response
     */
    public function edit(KerasModel $kerasModel)
    {
        return response()->json(['status' => 200, 'kerasModel' => $kerasModel]);
    }

    /**
     * Update the specified resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @param  \App\Models\KerasModel  $kerasModel
     * @return \Illuminate\Http\Response
     */
    public function update(Request $request, KerasModel $kerasModel)
    {
        $kerasModel->update($request->all());
        return response()->json(['status' => 200, 'kerasModel' => $kerasModel]);
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  \App\Models\KerasModel  $kerasModel
     * @return \Illuminate\Http\Response
     */
    public function destroy(KerasModel $kerasModel)
    {
        if (config('env.appEnv') == "local") {
            $file = public_path("/kerasModels/" . $kerasModel->kerasModelFile);
            File::delete($file);
        } elseif (config('env.appEnv') == "production") {
            Storage::disk('s3')->delete('kerasModels/' . $kerasModel->kerasModelFile);
        }
        $kerasModel->delete();
        return response()->json(["status" => 204]);
    }

    public function destroyMany(Request $request)
    {
        $selectedKerasModelIds = $request->selectedKerasModelIds;
        $kerasModelsToDelete = KerasModel::whereIn('id', $selectedKerasModelIds)->get();
        foreach ($kerasModelsToDelete as $kerasModelToDelete) {
            if (config('env.appEnv') == "local") {
                $file = public_path("/kerasModels/" . $kerasModelToDelete->kerasModelFile);
                File::delete($file);
            } elseif (config('env.appEnv') == "production") {
                Storage::disk('s3')->delete('kerasModels/' . $kerasModelToDelete->kerasModelFile);
            }
            $kerasModelToDelete->delete();
        }
        return response()->json(['status' => 204, 'kerasModel' => $kerasModelsToDelete]);
    }

    public function downloadFile(KerasModel $kerasModel)
    {
        if (config('env.appEnv') == "local") {
            $file = public_path("/kerasModels/" . $kerasModel->kerasModelFile);
            return response()->download($file);
        } elseif (config('env.appEnv') == "production") {
            $file = Storage::disk('s3')->get('kerasModels/' . $kerasModel->kerasModelFile);
            $headers = [
                'Content-Type' => 'zip',
                'Content-Description' => 'File Transfer',
                'Content-Disposition' => "attachment; filename={$kerasModel->kerasModelFile}",
                'filename' => $kerasModel->kerasModelFile
            ];
            return response($file, 200, $headers);;
        }
    }
}
