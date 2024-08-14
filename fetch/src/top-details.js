// import necessary modules
const axios = require('axios')
const fs = require('fs')

// import cities name, weeks dates, authorisation tokens
let cities = require('../data/cities')
let weeks = require('../data/weeks')
let token = require('../data/token')

// top 10 songs
let noTrackNeed = 10

// for etd. time
let timeTaken = 0
let complete = 0


async function track(city, week) {

    const data = require(`../../assets/city/${city}${week}.json`)

    // noTrackNeed = data.entries.length

    let newDataPromises = []
    for (let i = 0; i < noTrackNeed; i++) {
        const element = data.entries[i]
        const artists = element.trackMetadata.artists.map((elm) => { return elm.name })
        try {
        // making request to API
            const fetchedData = await axios.get(`https://api.spotify.com/v1/audio-features/${element.trackMetadata.trackUri.split(':')[2]}`, {
                headers: {
                    authorization: token.apiToken,
                }
            })
            // selecting necessary data field
            newDataPromises.push({
                city: city,
                week: week,
                trackName: element.trackMetadata.trackName,
                trackID: element.trackMetadata.trackUri.split(':')[2],
                artists: artists.toString(),
                ...element.chartEntryData,
                ...fetchedData.data
            })
        } catch (error) {
            console.log(`${city}${week} fetch error   XXXX`)
        }
    }
    try {
        // saving the data into json file
        await fs.promises.writeFile(`./assets/track/${city}${week}.json`, JSON.stringify(newDataPromises), 'utf-8')
    } catch (error) {
        console.log(`${city}${week} save error   XXXX`)
    }
}

// looping over all cities and weeks and calulating est. time
async function main() {
    for (let j = 0; j < cities.length; j++) {
        for (let k = 0; k < weeks.length; k++) {
            try {
                timeTaken = timeTaken - Date.now()
                await track(cities[j].toLowerCase(), weeks[k])
            } catch (err) {
                complete = complete - 1
                console.log(`${cities[j].toLowerCase()} ${weeks[k]} Error---------------`)
            } finally {
                complete++
                timeTaken = timeTaken + Date.now()
                console.log(`${cities[j]}${weeks[k]} Saved    [est.${Math.round(
                    (timeTaken / (complete * 60000)) * ((cities.length * weeks.length) - complete)
                )}min ${complete}/${cities.length * weeks.length}]`)
            }
        }
    }
}

// start
main()

//test
// track("kolkata","2023-03-09")