<?php

namespace Database\Factories;

use App\Models\KerasModel;
use Illuminate\Database\Eloquent\Factories\Factory;

class KerasModelFactory extends Factory
{
    /**
     * The name of the factory's corresponding model.
     *
     * @var string
     */
    protected $model = KerasModel::class;

    /**
     * Define the model's default state.
     *
     * @return array
     */
    public function definition()
    {
        return [
            'file_name' => $this->faker->sentence(1),
            'description' => $this->faker->sentence(3),
        ];
    }
}
