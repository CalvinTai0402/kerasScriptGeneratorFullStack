<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class KerasModel extends Model
{
    use HasFactory;
    protected $table = 'keras_models';
    protected $fillable = [
        'file_name',
        'description',
        'kerasModelFile'
    ];

    public function scopeFileName($query, $filter)
    {
        if (!is_null($filter)) {
            return $query->where('file_name', 'LIKE', '%' . $filter . '%');
        }

        return $query;
    }

    public function scopeDescription($query, $filter)
    {
        if (!is_null($filter)) {
            return $query->orWhere('description', 'LIKE', '%' . $filter . '%');
        }

        return $query;
    }

    public function scopeOrder($query, $field, $order)
    {
        if (!is_null($field)) {
            return $query->orderBy($field, $order);
        }

        return $query;
    }

    public function scopeSkipPage($query, $toSkip)
    {
        if ($toSkip != 0) {
            return $query->skip($toSkip);
        }

        return $query;
    }
}
